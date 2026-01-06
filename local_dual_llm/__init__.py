from .engine import QwenEngine
from .config import LocalDualLLMConfig
from .pipeline import inference_pdf, batched_inference, recursive_batched_inference

__version__ = "1.0.2"
__author__ = "Strategion"
__email__ = "development@strategion.de"

__all__ = ["QwenEngine", "LocalDualLLMConfig", "inference_pdf", "batched_inference", "recursive_batched_inference"]