import os
from dataclasses import dataclass

@dataclass
class LocalDualLLMConfig:
    # Default Directories. If not present locally, they will be created during the first runtime.
    dataset_dir: str = "./dataset" # Temporary folder. Holds the pages contained within each pdf during runtime. Is cleared automatically.
    results_dir: str = "./results" # Contains the results after running functions from the pipeline.
    cache_dir: str = os.environ.get("HF_HOME", "./Qwen2.5VL_cache") # Holds model weights for a specific model-id. Auto-downloads for a specific model on the first run.
    
    # Image Settings
    min_pixels: int = 1024 * 28 * 28
    max_pixels: int = 2048 * 28 * 28
    
    # Generation Settings
    gen_max_new_tokens_ocr: int = 2086
    gen_max_new_tokens_report: int = 2286
    
    # Model Settings
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "auto" # device mapping, "cuda" if present or else "cpu" 

    @property
    def debug_log_dir(self):
        return os.path.join(self.results_dir, "debug_logs")