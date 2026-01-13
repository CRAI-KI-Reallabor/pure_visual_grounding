from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

@dataclass
class PipelineConfig:
    # paths
    dots_model_path: PathLike
    image_folder: PathLike
    pdf_path: Optional[PathLike]
    crops_dir: PathLike
    reports_dir: PathLike

    # pdf rendering
    pdf_dpi: int = 200
    max_long_edge_px: Optional[int] = None

    # dots params
    padding: int = 12
    max_side: int = 1280
    batch_size: int = 18
    base_max_new_tokens: int = 12000
    extra_max_new_tokens: int = 15000

    # gemma params
    gemma_model_id: str = "google/gemma-3n-e4b-it"
    crop_upscale: float = 2.0
    summary_max_tokens: int = 2048  # Tokens for summary generation (increased for comprehensive summaries)

    # runtime/model loading
    attn_impl: str = "flash_attention_2"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"  # "bfloat16" or "float16"

    # manifest options
    save_manifest: bool = False
    manifest_filename: str = "picture_regions_manifest_summary.json"

    # logging
    verbose: bool = False
