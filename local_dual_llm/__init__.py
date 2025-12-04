from .engine import QwenEngine
from .config import QwenConfig
from .pipeline import inference_pdf, batched_inference

__version__ = "1.0.2"
__author__ = "Strategion"
__email__ = "development@strategion.de"

__all__ = ["QwenEngine", "LocalDualLLMConfig", "inference_pdf", "batched_inference"]