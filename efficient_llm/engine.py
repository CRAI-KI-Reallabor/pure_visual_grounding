import gc
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, Gemma3nForConditionalGeneration

from qwen_vl_utils import process_vision_info

from .utils import extract_json_array


class DotsLayoutEngine:
    def __init__(
        self,
        model_path: Union[str, Path],
        attn_impl: str = "flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map: str = "auto",
    ):
        self.model_path = str(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        torch.set_grad_enabled(False)

    def close(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        torch.cuda.empty_cache()


def dots_regenerate_with_more_tokens(
    engine: DotsLayoutEngine,
    img: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    try:
        conv = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }]
        messages = [conv]
        text = engine.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = engine.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(engine.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = engine.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=1.0,
                use_cache=True,
            )

        trimmed = out_ids[0, inputs["input_ids"][0].shape[0]:]
        decoded = engine.processor.decode(trimmed, skip_special_tokens=True)
        parsed = extract_json_array(decoded)

        del inputs, image_inputs, video_inputs, out_ids, trimmed
        torch.cuda.empty_cache()
        gc.collect()

        return parsed

    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return []
    except Exception:
        torch.cuda.empty_cache()
        gc.collect()
        return []


class GemmaCropOcrEngine:
    def __init__(self, model_id: str = "google/gemma-3n-e4b-it", dtype=torch.bfloat16, device_map: str = "auto"):
        self.model_id = model_id
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, device_map=device_map, torch_dtype=dtype
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def close(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        torch.cuda.empty_cache()

    def ocr_crop_plaintext(self, img: Image.Image, max_new_tokens: int = 512) -> str:
        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (
                        "You are an industrial-technical OCR assistant. "
                        "Your task is ONLY to perform OCR for the provided cropped technical drawing region. "
                        "Return PLAIN TEXT ONLY. No comments, no translation, no extra words. "
                        "Extract ONLY the exact text content visible within the cropped images. "
                        "Absolutely NO guessing, inferring, paraphrasing, or fabrication is allowed."
                    ),
                }],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Extract ALL visible text inside this crop. Plain text only."},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = gen[0, inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(generated, skip_special_tokens=True)
        return (text or "").strip()
