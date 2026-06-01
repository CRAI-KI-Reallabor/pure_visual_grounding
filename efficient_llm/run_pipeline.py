# efficient_llm/run_pipeline.py

import argparse
import os
from pathlib import Path

# Force expansive segments to avoid fragmentation OOM on high-VRAM cards with varying tensor sizes
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from efficient_llm.config import PipelineConfig
from efficient_llm.pipeline import run_pipeline, run_pipeline_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Efficient DOTS + Gemma OCR pipeline"
    )

    parser.add_argument("--dots-model", required=True, help="Path to DOTS model")
    parser.add_argument("--image-folder", help="Folder containing PNGs (optional if pdf provided)")
    parser.add_argument("--pdf", default=None, help="Optional PDF path (single document mode)")
    parser.add_argument("--pdfs", nargs="+", default=None, help="Multiple PDF paths for batch mode — DOTS and Gemma load only once across all documents.")
    parser.add_argument("--pdf-folder", default=None, help="Folder containing PDFs to batch process. All .pdf files in the folder are processed with shared model loads.")
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
    parser.add_argument("--ocr-max-new-tokens", type=int, default=512, help="Max tokens for Gemma crop OCR. Raise (e.g. 1536) for dense diagrams where some labels get truncated.")

    # Runtime params
    parser.add_argument("--attn-impl", default="flash_attention_2", help="Attention implementation (e.g. flash_attention_2, eager, sdpa)")
    parser.add_argument("--torch-dtype", default="bfloat16", help="Torch dtype (bfloat16 or float16)")
    parser.add_argument("--device-map", default="auto", help="Device map (auto, cuda, cpu)")

    parser.add_argument(
        "--generate-summary",
        action="store_true",
        help="Enable Gemma-based page summary generation (disabled by default for performance).",
    )
    parser.add_argument(
        "--generate-picture-summary",
        action="store_true",
        help="Enable per-crop semantic description of engineering drawings (disabled by default for performance).",
    )
    parser.add_argument(
        "--no-tiling",
        action="store_true",
        help="Disable tiled OCR for large crops. Use when tiling introduces noise in OCR results.",
    )
    parser.add_argument(
        "--dots-on-crops",
        action="store_true",
        help="Run DOTS OCR on each picture crop to extract embedded text labels inside diagrams.",
    )
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


def _build_cfg(args, pdf_path, image_folder, base_name):
    """Build a PipelineConfig with smart defaults for the given input."""
    if not image_folder:
        image_folder = Path(f"./{base_name}_pngs")

    crops_dir = Path(args.crops_dir) if args.crops_dir else Path(f"./{base_name}_crops")
    reports_dir = Path(args.reports_dir) if args.reports_dir else Path(f"./{base_name}_reports")

    return PipelineConfig(
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
        generate_summary=args.generate_summary,
        generate_picture_summary=args.generate_picture_summary,
        use_tiling=not args.no_tiling,
        dots_on_crops=args.dots_on_crops,
        ocr_max_new_tokens=args.ocr_max_new_tokens,
        save_manifest=args.save_manifest,
        verbose=args.verbose,
        attn_impl=args.attn_impl,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )


def main():
    args = parse_args()

    # Resolve batch input — either explicit list (--pdfs) or folder discovery (--pdf-folder)
    batch_pdfs = None
    if args.pdfs and args.pdf_folder:
        raise ValueError("Use either --pdfs or --pdf-folder, not both.")
    if args.pdfs:
        batch_pdfs = [Path(p) for p in args.pdfs]
    elif args.pdf_folder:
        folder = Path(args.pdf_folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"--pdf-folder is not a directory: {folder}")
        batch_pdfs = sorted(folder.glob("*.pdf"))
        if not batch_pdfs:
            raise FileNotFoundError(f"No .pdf files found in {folder}")
        print(f"[BATCH] Discovered {len(batch_pdfs)} PDF(s) in {folder}")

    # Batch mode — multiple PDFs share a single load of DOTS and Gemma
    if batch_pdfs:
        if args.crops_dir or args.reports_dir or args.image_folder:
            raise ValueError("--crops-dir, --reports-dir, --image-folder are not supported in batch mode (auto-derived per document).")

        cfgs = []
        for pdf_path in batch_pdfs:
            base_name = pdf_path.stem
            cfgs.append(_build_cfg(args, pdf_path, None, base_name))

        out_paths = run_pipeline_batch(cfgs)
        print(f"[DONE] Processed {len(out_paths)} document(s):")
        for p in out_paths:
            print(f"  - {p}")
        return

    # Single document mode
    pdf_path = Path(args.pdf) if args.pdf else None
    image_folder = Path(args.image_folder) if args.image_folder else None

    if not pdf_path and not image_folder:
        raise ValueError("You must provide either --pdf, --pdfs, or --image-folder.")

    # Determine a base name for defaults
    if pdf_path:
        base_name = pdf_path.stem
    else:
        base_name = image_folder.name

    cfg = _build_cfg(args, pdf_path, image_folder, base_name)

    print(f"Configuration:")
    print(f"  PDF: {pdf_path}")
    print(f"  Images: {cfg.image_folder}")
    print(f"  Crops: {cfg.crops_dir}")
    print(f"  Reports: {cfg.reports_dir}")

    out_path = run_pipeline(cfg)
    print(f"[DONE] Final report written to: {out_path}")


if __name__ == "__main__":
    main()
