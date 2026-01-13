import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import torch
from PIL import Image

from qwen_vl_utils import process_vision_info

from .config import PipelineConfig
from .engine import DotsLayoutEngine, GemmaCropOcrEngine, dots_regenerate_with_more_tokens
from .prompts import DOTS_LAYOUT_PROMPT
from .utils import (
    convert_pdf_to_images,
    extract_json_array,
    parse_ocr_file,
    extract_picture_regions,
    build_picture_manifest,
    upscale_image,
)


def run_dots_layout_on_folder(
    engine: DotsLayoutEngine,
    image_folder: Union[str, Path],
    prompt: str = DOTS_LAYOUT_PROMPT,
    batch_size: int = 18,
    base_max_new_tokens: int = 12000,
    extra_max_new_tokens: int = 12000,
) -> Dict[str, List[Dict[str, Any]]]:
    image_folder = Path(image_folder)
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".png")])
    if not image_files:
        raise FileNotFoundError(f"No PNG files found in: {image_folder}")

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    n_images = len(image_files)

    for batch_start_idx in range(0, n_images, batch_size):
        batch_files = image_files[batch_start_idx:batch_start_idx + batch_size]

        pil_images: List[Image.Image] = []
        valid_files: List[str] = []
        for img_file in batch_files:
            p = image_folder / img_file
            try:
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    pil_images.append(img.copy())
                    valid_files.append(img_file)
            except Exception:
                all_results[img_file] = []

        if not pil_images:
            continue

        batch_messages = []
        for img in pil_images:
            conv = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }]
            batch_messages.append(conv)

        text = engine.processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = engine.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(engine.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        try:
            with torch.inference_mode():
                out_ids = engine.model.generate(
                    **inputs,
                    max_new_tokens=base_max_new_tokens,
                    do_sample=False,
                    top_p=1.0,
                    use_cache=True,
                )

            trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
            decoded = engine.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for fname, raw_text, img in zip(valid_files, decoded, pil_images):
                parsed = extract_json_array(raw_text)
                if not parsed:
                    parsed = dots_regenerate_with_more_tokens(engine, img, prompt, extra_max_new_tokens)
                all_results[fname] = parsed

        except torch.OutOfMemoryError:
            for fname in valid_files:
                all_results[fname] = []
            torch.cuda.empty_cache()
        finally:
            del inputs, image_inputs, video_inputs
            torch.cuda.empty_cache()

    return all_results


def build_final_reports(
    manifest: Dict[str, Any],
    ocr_map: Dict[str, List[Dict[str, Any]]],
    gemma_engine: GemmaCropOcrEngine,
    crop_upscale: float = 2.0,
    image_folder: Union[str, Path] = None,
) -> Dict[str, Any]:
    """
    Build final reports with structured output format.
    
    This function structures the OCR results and generates page summaries using Gemma.
    The output format matches the structure used by local_dual_llm for consistency.
    
    Args:
        manifest: Picture regions manifest from build_picture_manifest
        ocr_map: OCR results from DOTS layout engine
        gemma_engine: Gemma engine for crop OCR and summary generation
        crop_upscale: Upscaling factor for cropped images
        image_folder: Path to folder containing full-page images (for summary generation)
    
    Returns:
        Dict with structure:
        {
            "pdf_name": str,
            "pages": [
                {
                    "page_image": str,
                    "OCR_Result": {
                        "ocr_pass_result": [...],  # DOTS layout results
                        "picture_ocr_result": [...]  # Gemma crop OCR results
                    },
                    "Generated_Report": {
                        "summary": str  # Gemma-generated page summary
                    },
                    "summary": str  # Convenience field (same as Generated_Report.summary)
                }
            ],
            "errors": []
        }
    """
    images_list = manifest.get("images", [])
    pages_reports: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    
    # Try to extract PDF name from first image name
    pdf_name = "document"
    if images_list and images_list[0].get("image_name"):
        first_img = images_list[0]["image_name"]
        # Extract base name without _page_X.png
        import re
        match = re.match(r'(.+?)_page_\d+\.png', first_img)
        if match:
            pdf_name = match.group(1)

    for page in images_list:
        image_name = page.get("image_name")
        picture_regions = page.get("picture_regions", [])

        ocr_pass_result = ocr_map.get(image_name, []) or ocr_map.get(str(image_name).lower(), [])

        picture_regions_sorted = sorted(
            picture_regions,
            key=lambda r: (r.get("order", 10**9), r.get("region_id", ""))
        )

        picture_ocr_result: List[Dict[str, Any]] = []

        for r in picture_regions_sorted:
            rid = r.get("region_id")
            bbox = r.get("bbox")
            crop_path = r.get("crop_image_path")

            if not crop_path or not Path(crop_path).exists():
                picture_ocr_result.append({"region_id": rid, "bbox": bbox, "text": ""})
                continue

            try:
                with Image.open(crop_path) as img:
                    img = img.convert("RGB")
                    if crop_upscale and crop_upscale != 1.0:
                        img = upscale_image(img, scale=float(crop_upscale))
                    txt = gemma_engine.ocr_crop_plaintext(img)

                picture_ocr_result.append({"region_id": rid, "bbox": bbox, "text": txt})
            except Exception as e:
                picture_ocr_result.append({"region_id": rid, "bbox": bbox, "text": ""})
                errors.append({"page": image_name, "region": rid, "error": str(e)})

        # Build OCR_Result (combining both passes)
        ocr_result = {
            "ocr_pass_result": ocr_pass_result,
            "picture_ocr_result": picture_ocr_result
        }

        # Generate summary using the full page image
        summary_text = ""
        try:
            if image_folder:
                full_page_path = Path(image_folder) / image_name
                if full_page_path.exists():
                    with Image.open(full_page_path) as full_img:
                        full_img = full_img.convert("RGB")
                        # Create OCR context for summary
                        ocr_context = json.dumps(ocr_result, ensure_ascii=False, indent=2)
                        summary_result = gemma_engine.generate_page_summary(
                            full_img, 
                            ocr_context=ocr_context,
                            max_new_tokens=2048  # Use increased token limit for comprehensive summaries
                        )
                        summary_text = summary_result.get("summary", "")
        except Exception as e:
            errors.append({"page": image_name, "error": f"Summary generation failed: {str(e)}"})
            summary_text = "Summary generation failed."

        # Build Generated_Report
        generated_report = {
            "summary": summary_text
        }

        pages_reports.append({
            "page_image": image_name,
            "OCR_Result": ocr_result,
            "Generated_Report": generated_report,
            "summary": summary_text  # Top-level summary for convenience
        })

    return {
        "pdf_name": pdf_name,
        "pages": pages_reports,
        "errors": errors
    }


def run_pipeline(cfg: PipelineConfig) -> Path:
    image_folder = Path(cfg.image_folder)
    crops_dir = Path(cfg.crops_dir)
    reports_dir = Path(cfg.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    image_folder.mkdir(parents=True, exist_ok=True)

    # Ensure PNGs exist
    pngs = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".png")])
    if not pngs:
        if cfg.pdf_path and Path(cfg.pdf_path).is_file():
            convert_pdf_to_images(cfg.pdf_path, image_folder)
        else:
            raise FileNotFoundError("No PNGs found and pdf_path is missing/invalid.")

    # dtype
    dtype = torch.bfloat16 if cfg.torch_dtype.lower() == "bfloat16" else torch.float16

    # 1) DOTS
    dots_engine = DotsLayoutEngine(
        model_path=cfg.dots_model_path,
        attn_impl=cfg.attn_impl,
        torch_dtype=dtype,
        device_map=cfg.device_map,
    )
    try:
        ocr_map = run_dots_layout_on_folder(
            engine=dots_engine,
            image_folder=image_folder,
            prompt=DOTS_LAYOUT_PROMPT,
            batch_size=cfg.batch_size,
            base_max_new_tokens=cfg.base_max_new_tokens,
            extra_max_new_tokens=cfg.extra_max_new_tokens,
        )
    finally:
        dots_engine.close()

    dots_out_path = image_folder / "dots_ocr_result_combined.json"
    with open(dots_out_path, "w", encoding="utf-8") as f:
        json.dump(ocr_map, f, ensure_ascii=False, indent=2)

    # 2) Crop picture regions
    parsed_map = parse_ocr_file(dots_out_path)
    picture_map = extract_picture_regions(parsed_map)
    manifest = build_picture_manifest(
        picture_map=picture_map,
        image_root=image_folder,
        crops_dir=crops_dir,
        padding=cfg.padding,
        do_crop=True,
    )

    if cfg.save_manifest:
        manifest_path = crops_dir.parent / cfg.manifest_filename
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 3) Gemma crop OCR + Summary generation
    gemma_engine = GemmaCropOcrEngine(model_id=cfg.gemma_model_id, dtype=dtype, device_map=cfg.device_map)
    try:
        final_reports = build_final_reports(
            manifest=manifest,
            ocr_map=parsed_map,
            gemma_engine=gemma_engine,
            crop_upscale=cfg.crop_upscale,
            image_folder=image_folder,  # Pass image_folder for summary generation
        )
    finally:
        gemma_engine.close()

    out_json_path = reports_dir / "final_image_description_reports_combined_batch_v2.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(final_reports, f, ensure_ascii=False, indent=2)

    return out_json_path
