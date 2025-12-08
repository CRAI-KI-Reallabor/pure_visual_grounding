import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from .config import LocalDualLLMConfig
import gc

class QwenEngine:
    def __init__(self, config: LocalDualLLMConfig = None, cache_dir: str = None, model_id: str = None):
        """
        Args:
            config: A LocalDualLLMConfig object. If None, defaults are used.
            cache_dir: specific override for cache location.
            model_id: specific override for model ID.
        """
        # 1. Setup Configuration
        self.cfg = config if config else LocalDualLLMConfig()
        
        # Overrides if provided directly
        target_cache = cache_dir if cache_dir else self.cfg.cache_dir
        target_model = model_id if model_id else self.cfg.model_id
        
        # Ensure cache exists
        if target_cache:
            os.makedirs(target_cache, exist_ok=True)

        self.device = self.cfg.device
        if self.device == "auto":
             self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"--- Loading {target_model} on {self.device}... ---")
        
        # 2. Load Processor (Auto-download logic handled by transformers)
        self.processor = AutoProcessor.from_pretrained(
            target_model, 
            cache_dir=target_cache
        )

        # 3. Load Model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            target_model,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="sdpa",
            cache_dir=target_cache,
        )
        print("--- Model Loaded Successfully ---")
        pass

    def run_inference(self, messages, max_new_tokens):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        trimmed = generated_ids[0, len(inputs.input_ids[0]):]
        return self.processor.decode(trimmed, skip_special_tokens=True)

    def close(self):
        print("--- Closing Engine and freeing memory ---")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        
        # Explicit garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    # Context Manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("--- Auto-Closing Engine and freeing memory ---")
        self.close()