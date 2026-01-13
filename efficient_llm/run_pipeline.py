# efficient_llm/run_pipeline.py

import argparse
import os
from pathlib import Path

# Force expansive segments to avoid fragmentation OOM on high-VRAM cards with varying tensor sizes
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from efficient_llm.config import PipelineConfig
from efficient_llm.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Efficient DOTS + Gemma OCR pipeline"
    )

    parser.add_argument("--dots-model", required=True, help="Path to DOTS model")
    parser.add_argument("--image-folder", help="Folder containing PNGs (optional if pdf provided)")
    parser.add_argument("--pdf", default=None, help="Optional PDF path")
    parser.add_argument("--crops-dir", help="Directory to store cropped images (optional)")
    parser.add_argument("--reports-dir", help="Directory to store final reports (optional)")

    parser.add_argument("--pdf-dpi", type=int, default=200, help="DPI used when rasterizing PDF pages (default: 200)")
    parser.add_argument("--max-long-edge-px", type=int, default=None, help="If set, downscale rendered pages so longest edge <= this many pixels")

    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--base-max-new-tokens", type=int, default=12000)
    parser.add_argument("--extra-max-new-tokens", type=int, default=15000)
    parser.add_argument("--gemma-model-id", default="google/gemma-3n-e4b-it")
    parser.add_argument("--crop-upscale", type=float, default=2.0)

    # Runtime params
    parser.add_argument("--attn-impl", default="flash_attention_2", help="Attention implementation (e.g. flash_attention_2, eager, sdpa)")
    parser.add_argument("--torch-dtype", default="bfloat16", help="Torch dtype (bfloat16 or float16)")
    parser.add_argument("--device-map", default="auto", help="Device map (auto, cuda, cpu)")

    parser.add_argument(
        "--save-manifest",
        action="store_true",
        help="Save picture region manifest summary JSON for debugging/reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging and progress bars.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Smart defaults logic
    pdf_path = Path(args.pdf) if args.pdf else None
    image_folder = Path(args.image_folder) if args.image_folder else None

    if not pdf_path and not image_folder:
        raise ValueError("You must provide either --pdf or --image-folder.")

    # Determine a base name for defaults
    if pdf_path:
        base_name = pdf_path.stem
    else:
        # image_folder must exist here
        base_name = image_folder.name

    # Apply defaults if missing
    if not image_folder:
        image_folder = Path(f"./{base_name}_pngs")
    
    crops_dir = Path(args.crops_dir) if args.crops_dir else Path(f"./{base_name}_crops")
    reports_dir = Path(args.reports_dir) if args.reports_dir else Path(f"./{base_name}_reports")

    print(f"Configuration:")
    print(f"  PDF: {pdf_path}")
    print(f"  Images: {image_folder}")
    print(f"  Crops: {crops_dir}")
    print(f"  Reports: {reports_dir}")

    cfg = PipelineConfig(
        dots_model_path=Path(args.dots_model),
        image_folder=image_folder,
        pdf_path=pdf_path,
        crops_dir=crops_dir,
        reports_dir=reports_dir,
        padding=args.padding,
        batch_size=args.batch_size,
        base_max_new_tokens=args.base_max_new_tokens,
        extra_max_new_tokens=args.extra_max_new_tokens,
        gemma_model_id=args.gemma_model_id,
        crop_upscale=args.crop_upscale,
        pdf_dpi=args.pdf_dpi,
        max_long_edge_px=args.max_long_edge_px,
        save_manifest=args.save_manifest,
        verbose=args.verbose,
        attn_impl=args.attn_impl,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )

    out_path = run_pipeline(cfg)
    print(f"[DONE] Final report written to: {out_path}")


if __name__ == "__main__":
    main()
