# efficient_llm/example_test_batch.py

import argparse
from pathlib import Path

from efficient_llm.config import PipelineConfig
from efficient_llm.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Solarlux DOTS + Gemma OCR pipeline"
    )

    parser.add_argument("--dots-model", required=True, help="Path to DOTS model")
    parser.add_argument("--image-folder", required=True, help="Folder containing PNGs")
    parser.add_argument("--pdf", default=None, help="Optional PDF path")
    parser.add_argument("--crops-dir", required=True, help="Directory to store cropped images")
    parser.add_argument("--reports-dir", required=True, help="Directory to store final reports")

    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--base-max-new-tokens", type=int, default=12000)
    parser.add_argument("--extra-max-new-tokens", type=int, default=15000)
    parser.add_argument("--gemma-model-id", default="google/gemma-3n-e4b-it")
    parser.add_argument("--crop-upscale", type=float, default=2.0)

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = PipelineConfig(
        dots_model_path=Path(args.dots_model),
        image_folder=Path(args.image_folder),
        pdf_path=Path(args.pdf) if args.pdf else None,
        crops_dir=Path(args.crops_dir),
        reports_dir=Path(args.reports_dir),
        padding=args.padding,
        batch_size=args.batch_size,
        base_max_new_tokens=args.base_max_new_tokens,
        extra_max_new_tokens=args.extra_max_new_tokens,
        gemma_model_id=args.gemma_model_id,
        crop_upscale=args.crop_upscale,
    )

    out_path = run_pipeline(cfg)
    print(f"[DONE] Final report written to: {out_path}")


if __name__ == "__main__":
    main()
