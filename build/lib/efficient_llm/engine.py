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
        Perform OCR on multiple cropped images with automatic GPU-aware batching.
        
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
                    # Get available GPU memory in GB
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    # Heuristic: adjust batch size based on GPU memory
                    if gpu_mem_gb >= 40:  # A100, H100
                        batch_size = 32
                    elif gpu_mem_gb >= 24:  # RTX 4090, A5000
                        batch_size = 16
                    elif gpu_mem_gb >= 16:  # RTX 4080, A4000
                        batch_size = 12
                    elif gpu_mem_gb >= 12:  # RTX 3090, 4070 Ti
                        batch_size = 8
                    elif gpu_mem_gb >= 8:  # RTX 3070, 4060 Ti
                        batch_size = 6
                    else:  # Smaller GPUs
                        batch_size = 4
                    
                    if self.verbose:
                        print(f"[GEMMA] Auto-detected GPU memory: {gpu_mem_gb:.1f}GB, using batch_size={batch_size}")
                else:
                    batch_size = 4  # CPU fallback
            except:
                batch_size = 8  # Safe default
        
        # System prompt used for all crops
        system_content = [{
            "type": "text",
            "text": (
                "You are an industrial-technical OCR assistant. "
                "Your task is ONLY to perform OCR for the provided cropped technical drawing region. "
                "Return PLAIN TEXT ONLY. No comments, no translation, no extra words. "
                "Extract ONLY the exact text content visible within the cropped images. "
                "Absolutely NO guessing, inferring, paraphrasing, or fabrication is allowed."
            ),
        }]
        
        all_results = []
        
        # Process images in chunks
        for chunk_start in range(0, len(images), batch_size):
            chunk_end = min(chunk_start + batch_size, len(images))
            chunk_images = images[chunk_start:chunk_end]
            
            if self.verbose:
                print(f"[GEMMA] Processing batch {chunk_start//batch_size + 1}: images {chunk_start+1}-{chunk_end}/{len(images)}")
            
            # Prepare batch messages for this chunk
            batch_messages = []
            for img in chunk_images:
                messages = [
                    {"role": "system", "content": system_content},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Extract ALL visible text inside this crop. Plain text only."},
                        ],
                    },
                ]
                batch_messages.append(messages)
            
            try:
                # Process this chunk
                inputs = self.processor.apply_chat_template(
                    batch_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,
                ).to(self.model.device)
                
                with torch.inference_mode():
                    gen = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                
                # Decode results for this chunk
                for i in range(len(chunk_images)):
                    generated = gen[i, inputs["input_ids"].shape[-1]:]
                    text = self.processor.decode(generated, skip_special_tokens=True)
                    all_results.append((text or "").strip())
                
                # Clean up
                del inputs, gen
                torch.cuda.empty_cache()
                
            except torch.OutOfMemoryError:
                # If OOM, fallback to processing one at a time for this chunk
                if self.verbose:
                    print(f"[GEMMA] OOM error, falling back to sequential processing for this batch")
                torch.cuda.empty_cache()
                
                for img in chunk_images:
                    try:
                        text = self.ocr_crop_plaintext(img, max_new_tokens)
                        all_results.append(text)
                    except:
                        all_results.append("")  # Fallback for failed individual images
        
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
