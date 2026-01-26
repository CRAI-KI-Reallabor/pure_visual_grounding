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
        verbose: bool = False,
    ):
        self.model_path = str(model_path)
        self.verbose = verbose
        if self.verbose:
            print(f"[DOTS] Loading model from: {self.model_path}")
            print(f"[DOTS] Requested Attention: {attn_impl}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()

        # CRITICAL: Verify which attention implementation is actually being used
        actual_attn = getattr(self.model.config, "_attn_implementation", "unknown")
        if self.verbose or actual_attn != attn_impl:
            print(f"[DOTS]   ACTUAL Attention Implementation: {actual_attn}")
            if actual_attn != attn_impl:
                print(f"[DOTS]   WARNING: Requested '{attn_impl}' but model is using '{actual_attn}'!")
                print(f"[DOTS]   This model may not support Flash Attention 2. Performance will be slower.")
                if attn_impl == "flash_attention_2":
                    print(f"[DOTS]  Try using --attn-impl sdpa (Scaled Dot Product Attention) as fallback")

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
    def __init__(self, model_id: str = "google/gemma-3n-e4b-it", dtype=torch.bfloat16, device_map: str = "auto", verbose: bool = False):
        self.verbose = verbose
        if self.verbose:
            print(f"[GEMMA] Loading model: {model_id}")
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

    def ocr_crop_plaintext_batch(self, images: List[Image.Image], max_new_tokens: int = 512, batch_size: int = None) -> List[str]:
        """
        Perform OCR on multiple cropped images with batch inference.
        
        Uses tensor batching for improved GPU utilization (~10-15% speedup).
        
        Args:
            images: List of PIL Images to process
            max_new_tokens: Maximum tokens per generation
            batch_size: Optional batch size. If None, automatically determined based on GPU memory
            
        Returns:
            List of extracted text strings (one per image)
        """
        if not images:
            return []
        
        # Auto-determine batch size based on available GPU memory
        if batch_size is None:
            try:
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    # Conservative batch sizes for vision models
                    if gpu_mem_gb >= 40:
                        batch_size = 8
                    elif gpu_mem_gb >= 24:
                        batch_size = 6
                    elif gpu_mem_gb >= 16:
                        batch_size = 4
                    elif gpu_mem_gb >= 12:
                        batch_size = 3
                    else:
                        batch_size = 2
                else:
                    batch_size = 1
            except:
                batch_size = 2
        
        # OCR prompts
        ocr_prompt = "Extract ALL visible text inside this crop. Plain text only."
        system_prompt = (
            "You are an industrial-technical OCR assistant. "
            "Your task is ONLY to perform OCR for the provided cropped technical drawing region. "
            "Return PLAIN TEXT ONLY. No comments, no translation, no extra words. "
            "Extract ONLY the exact text content visible within the cropped images. "
            "Absolutely NO guessing, inferring, paraphrasing, or fabrication is allowed."
        )
        
        all_results = []
        
        # Process images in chunks
        for chunk_start in range(0, len(images), batch_size):
            chunk_end = min(chunk_start + batch_size, len(images))
            chunk_images = images[chunk_start:chunk_end]
            
            try:
                # Process each image individually and collect inputs
                all_input_ids = []
                all_attention_masks = []
                all_pixel_values = []
                
                for img in chunk_images:
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": ocr_prompt},
                            ],
                        },
                    ]
                    
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                    
                    all_input_ids.append(inputs["input_ids"])
                    all_attention_masks.append(inputs["attention_mask"])
                    if "pixel_values" in inputs:
                        all_pixel_values.append(inputs["pixel_values"])
                
                # Pad and batch input_ids and attention_masks
                max_len = max(t.shape[-1] for t in all_input_ids)
                pad_token_id = self.processor.tokenizer.pad_token_id or 0
                
                padded_input_ids = []
                padded_attention_masks = []
                for ids, mask in zip(all_input_ids, all_attention_masks):
                    pad_len = max_len - ids.shape[-1]
                    if pad_len > 0:
                        # Left padding for decoder models
                        ids = torch.cat([torch.full((1, pad_len), pad_token_id, dtype=ids.dtype), ids], dim=-1)
                        mask = torch.cat([torch.zeros((1, pad_len), dtype=mask.dtype), mask], dim=-1)
                    padded_input_ids.append(ids)
                    padded_attention_masks.append(mask)
                
                batched_input_ids = torch.cat(padded_input_ids, dim=0).to(self.model.device)
                batched_attention_mask = torch.cat(padded_attention_masks, dim=0).to(self.model.device)
                
                batched_inputs = {
                    "input_ids": batched_input_ids,
                    "attention_mask": batched_attention_mask,
                }
                
                if all_pixel_values:
                    batched_inputs["pixel_values"] = torch.cat(all_pixel_values, dim=0).to(self.model.device)
                
                # Generate for entire batch at once
                with torch.inference_mode():
                    gen = self.model.generate(
                        **batched_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                    )
                
                # Decode results
                for i in range(len(chunk_images)):
                    generated = gen[i, max_len:]
                    text = self.processor.decode(generated, skip_special_tokens=True)
                    all_results.append((text or "").strip())
                
                # Clean up
                del batched_inputs, gen, all_input_ids, all_attention_masks, all_pixel_values
                torch.cuda.empty_cache()
                
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Fallback to sequential
                for img in chunk_images:
                    try:
                        text = self.ocr_crop_plaintext(img, max_new_tokens)
                        all_results.append(text)
                    except:
                        all_results.append("")
                        
            except Exception:
                torch.cuda.empty_cache()
                # Fallback to sequential
                for img in chunk_images:
                    try:
                        text = self.ocr_crop_plaintext(img, max_new_tokens)
                        all_results.append(text)
                    except:
                        all_results.append("")
        
        return all_results

    def generate_page_summary(self, img: Image.Image, ocr_context: str = "", max_new_tokens: int = 2048) -> Dict[str, Any]:
        """
        Concatenate OCR text for full page (no model generation for better performance).
        
        Args:
            img: Full page PIL Image (not used, kept for API compatibility)
            ocr_context: OCR text context to concatenate
            max_new_tokens: Not used, kept for API compatibility
            
        Returns:
            Dictionary with 'summary' key containing the concatenated OCR text
        """
        # Simply return the concatenated OCR context for better RAG performance
        summary_text = ocr_context.strip() if ocr_context else "No text content extracted."
        return {"summary": summary_text}
