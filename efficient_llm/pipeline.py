import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from tqdm import tqdm

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
    verbose: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    image_folder = Path(image_folder)
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".png")])
    if not image_files:
        raise FileNotFoundError(f"No PNG files found in: {image_folder}")

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    n_images = len(image_files)
    
    # Timing tracking
    total_start = time.time()
    total_load_time = total_pre_time = total_gen_time = total_post_time = 0.0

    if verbose:
        print(f"\n[DOTS] Processing {n_images} images in batches of {batch_size}...")

    for batch_start_idx in range(0, n_images, batch_size):
        batch_files = image_files[batch_start_idx:batch_start_idx + batch_size]
        batch_num = batch_start_idx // batch_size + 1
        total_batches = (n_images + batch_size - 1) // batch_size
        
        if verbose:
            print(f"\n[DOTS] Batch {batch_num}/{total_batches}: {batch_files}")
        
        t_batch_start = time.time()

        # Load & resize
        t0 = time.time()
        pil_images: List[Image.Image] = []
        valid_files: List[str] = []
        for img_file in batch_files:
            p = image_folder / img_file
            try:
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    pil_images.append(img.copy())
                    valid_files.append(img_file)
            except Exception as e:
                if verbose:
                    print(f"[DOTS] WARN: failed to load {img_file}: {e}")
                all_results[img_file] = []
        t1 = time.time()
        total_load_time += (t1 - t0)
        if verbose:
            print(f"[DOTS]   Load time: {t1 - t0:.2f}s")

        if not pil_images:
            continue

        # Preprocess
        t2 = time.time()
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
        t3 = time.time()
        total_pre_time += (t3 - t2)
        if verbose:
            print(f"[DOTS]   Preprocess time: {t3 - t2:.2f}s")

        # Generate
        t4 = time.time()
        if verbose:
            print(f"[DOTS]   Generating... (max_tokens={base_max_new_tokens})")
        try:
            with torch.inference_mode():
                out_ids = engine.model.generate(
                    **inputs,
                    max_new_tokens=base_max_new_tokens,
                    do_sample=False,
                    top_p=1.0,
                    use_cache=True,
                )
            t5 = time.time()
            total_gen_time += (t5 - t4)
            if verbose:
                print(f"[DOTS]   Generation time: {t5 - t4:.2f}s")

            # Postprocess
            t6 = time.time()
            trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
            decoded = engine.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for fname, raw_text, img in zip(valid_files, decoded, pil_images):
                parsed = extract_json_array(raw_text)
                if not parsed:
                    if verbose:
                        print(f"[DOTS]   JSON parse failed for {fname}; retrying with {extra_max_new_tokens} tokens")
                    parsed = dots_regenerate_with_more_tokens(engine, img, prompt, extra_max_new_tokens)
                all_results[fname] = parsed
            
            t7 = time.time()
            total_post_time += (t7 - t6)
            if verbose:
                print(f"[DOTS]   Postprocess time: {t7 - t6:.2f}s")

        except torch.OutOfMemoryError:
            t5 = time.time()
            total_gen_time += (t5 - t4)
            if verbose:
                print(f"[DOTS]   WARNING: OOM: skipping batch; consider lowering batch_size")
            for fname in valid_files:
                all_results[fname] = []
            torch.cuda.empty_cache()
        finally:
            del inputs, image_inputs, video_inputs
            if 'out_ids' in locals():
                del out_ids
            if 'trimmed' in locals():
                del trimmed
            torch.cuda.empty_cache()
        
        t_batch_end = time.time()
        if verbose:
            print(f"[DOTS]   Batch total: {t_batch_end - t_batch_start:.2f}s")

    total_end = time.time()
    total_time = total_end - total_start
    n = max(n_images, 1)
    
    timing_stats = {
        "total_images": n_images,
        "total_time": total_time,
        "avg_time_per_image": total_time / n,
        "total_load_time": total_load_time,
        "total_pre_time": total_pre_time,
        "total_gen_time": total_gen_time,
        "total_post_time": total_post_time,
    }
    
    if verbose:
        print(f"\n[DOTS] Completed {n_images} images in {total_time:.2f}s | avg {total_time/n:.2f}s/image")
        print(f"[DOTS] Avg breakdown per image:")
        print(f"  load+resize: {total_load_time/n:.2f}s")
        print(f"  preprocess : {total_pre_time/n:.2f}s")
        print(f"  generate   : {total_gen_time/n:.2f}s")
        print(f"  postprocess: {total_post_time/n:.2f}s")

    return all_results, timing_stats


def build_final_reports(
    manifest: Dict[str, Any],
    ocr_map: Dict[str, List[Dict[str, Any]]],
    gemma_engine: GemmaCropOcrEngine,
    crop_upscale: float = 2.0,
    image_folder: Union[str, Path] = None,
    summary_max_tokens: int = 2048,
    verbose: bool = False,
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
    
    # Timing tracking for Gemma
    total_gemma_start = time.time()
    total_crop_ocr_time = 0.0
    total_summary_time = 0.0
    total_crops_processed = 0

    if verbose:
        print(f"\n[GEMMA] Processing {len(images_list)} pages for crop OCR and summaries...")

    for page_idx, page in enumerate(images_list, 1):
        image_name = page.get("image_name")
        if verbose:
            print(f"\n[GEMMA] Page {page_idx}/{len(images_list)}: {image_name}")
        picture_regions = page.get("picture_regions", [])

        ocr_pass_result = ocr_map.get(image_name, []) or ocr_map.get(str(image_name).lower(), [])

        picture_regions_sorted = sorted(
            picture_regions,
            key=lambda r: (r.get("order", 10**9), r.get("region_id", ""))
        )

        picture_ocr_result: List[Dict[str, Any]] = []
        
        if verbose and len(picture_regions_sorted) > 0:
            print(f"[GEMMA]   Processing {len(picture_regions_sorted)} crop regions...")

        for crop_idx, r in enumerate(picture_regions_sorted, 1):
            rid = r.get("region_id")
            bbox = r.get("bbox")
            crop_path = r.get("crop_image_path")

            if not crop_path or not Path(crop_path).exists():
                picture_ocr_result.append({"region_id": rid, "bbox": bbox, "text": ""})
                continue

            try:
                with Image.open(crop_path) as img:
                    if verbose:
                        print(f"[GEMMA]     Crop {crop_idx}/{len(picture_regions_sorted)}: {Path(crop_path).name} size={img.size} region_id={rid}")
                    img = img.convert("RGB")
                    if crop_upscale and crop_upscale != 1.0:
                        img = upscale_image(img, scale=float(crop_upscale))
                    
                    t_crop_start = time.time()
                    txt = gemma_engine.ocr_crop_plaintext(img)
                    t_crop_end = time.time()
                    total_crop_ocr_time += (t_crop_end - t_crop_start)
                    total_crops_processed += 1

                picture_ocr_result.append({"region_id": rid, "bbox": bbox, "text": txt})
            except Exception as e:
                picture_ocr_result.append({"region_id": rid, "bbox": bbox, "text": ""})
                errors.append({"page": image_name, "region": rid, "error": str(e)})
                if verbose:
                    print(f"[GEMMA]     WARNING: Error processing crop {rid}: {e}")

        # Build OCR_Result (combining both passes)
        ocr_result = {
            "ocr_pass_result": ocr_pass_result,
            "picture_ocr_result": picture_ocr_result,
        }

        # Page summary using full page image (if available)
        summary_text = ""
        full_page_path = Path(image_folder) / image_name if image_folder else None
        if full_page_path and full_page_path.exists():
            try:
                if verbose:
                    print(f"[GEMMA]   Generating page summary...")
                t_summary_start = time.time()
                with Image.open(full_page_path) as full_img:
                    full_img = full_img.convert("RGB")
                    ocr_context = json.dumps(ocr_result, ensure_ascii=False, indent=2)
                    summary_result = gemma_engine.generate_page_summary(
                        full_img,
                        ocr_context=ocr_context,
                        max_new_tokens=summary_max_tokens,
                    )
                    summary_text = summary_result.get("summary", "")
                t_summary_end = time.time()
                total_summary_time += (t_summary_end - t_summary_start)
                if verbose:
                    print(f"[GEMMA]   Summary generated ({len(summary_text)} chars) in {t_summary_end - t_summary_start:.2f}s")
            except Exception as e:
                errors.append({"page": image_name, "error": f"Summary generation failed: {str(e)}"})
                summary_text = "Summary generation failed."
                if verbose:
                    print(f"[GEMMA]   WARNING: Summary generation failed: {e}")

        generated_report = {"summary": summary_text}

        pages_reports.append({
            "page_image": image_name,
            "OCR_Result": ocr_result,
            "Generated_Report": generated_report,
            "summary": summary_text,
        })

        if verbose:
            print(f"[GEMMA]   Page {image_name}: {len(picture_ocr_result)} crops processed")

    total_gemma_end = time.time()
    total_gemma_time = total_gemma_end - total_gemma_start
    n_pages = max(len(images_list), 1)
    n_crops = max(total_crops_processed, 1)
    
    timing_stats = {
        "total_pages": len(images_list),
        "total_crops": total_crops_processed,
        "total_time": total_gemma_time,
        "total_crop_ocr_time": total_crop_ocr_time,
        "total_summary_time": total_summary_time,
        "avg_time_per_page": total_gemma_time / n_pages,
        "avg_crop_ocr_time": total_crop_ocr_time / n_crops if total_crops_processed > 0 else 0,
        "avg_summary_time": total_summary_time / n_pages,
    }
    
    if verbose:
        print(f"\n[GEMMA] Completed all {len(images_list)} pages in {total_gemma_time:.2f}s")
        print(f"[GEMMA] Avg breakdown per page:")
        print(f"  total time   : {total_gemma_time/n_pages:.2f}s")
        print(f"  crop OCR     : {total_crop_ocr_time/n_pages:.2f}s ({total_crops_processed} crops total, avg {total_crop_ocr_time/n_crops:.2f}s/crop)")
        print(f"  summary gen  : {total_summary_time/n_pages:.2f}s")
        if errors:
            print(f"[GEMMA] WARNING: {len(errors)} errors encountered")

    report = {
        "pdf_name": pdf_name,
        "pages": pages_reports,
        "errors": errors,
    }
    
    return report, timing_stats


def run_pipeline(cfg: PipelineConfig) -> Path:
    image_folder = Path(cfg.image_folder)
    crops_dir = Path(cfg.crops_dir)
    reports_dir = Path(cfg.reports_dir)

    image_folder.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".png")])
    if not pngs:
        if cfg.pdf_path:
            if cfg.verbose:
                print(f"[PDF] Converting PDF to PNGs: {cfg.pdf_path} (dpi={cfg.pdf_dpi}, max_edge={cfg.max_long_edge_px})")
            convert_pdf_to_images(
                cfg.pdf_path,
                image_folder,
                dpi=cfg.pdf_dpi,
                max_long_edge_px=cfg.max_long_edge_px,
            )
            pngs = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".png")])
            if cfg.verbose:
                print(f"[PDF] Converted {len(pngs)} pages")
        else:
            raise FileNotFoundError("No PNGs found and pdf_path is missing/invalid.")
    elif cfg.verbose:
        print(f"[INFO] Found {len(pngs)} existing PNG files")

    # dtype
    dtype = torch.bfloat16 if cfg.torch_dtype.lower() == "bfloat16" else torch.float16

    # 1) DOTS
    if cfg.verbose:
        print(f"\n{'='*60}")
        print(f"STEP 1: DOTS Layout OCR")
        print(f"{'='*60}")
    dots_engine = DotsLayoutEngine(
        model_path=cfg.dots_model_path,
        attn_impl=cfg.attn_impl,
        torch_dtype=dtype,
        device_map=cfg.device_map,
        verbose=cfg.verbose,
    )
    try:
        ocr_map, dots_timing = run_dots_layout_on_folder(
            engine=dots_engine,
            image_folder=image_folder,
            prompt=DOTS_LAYOUT_PROMPT,
            batch_size=cfg.batch_size,
            base_max_new_tokens=cfg.base_max_new_tokens,
            extra_max_new_tokens=cfg.extra_max_new_tokens,
            verbose=cfg.verbose,
        )
    finally:
        dots_engine.close()

    if cfg.verbose:
        print(f"\n{'='*60}")
        print(f"STEP 2: Extract Picture Regions")
        print(f"{'='*60}")

    parsed_map = ocr_map
    picture_map = extract_picture_regions(parsed_map)
    manifest = build_picture_manifest(
        picture_map=picture_map,
        image_root=image_folder,
        crops_dir=crops_dir,
        padding=cfg.padding,
        do_crop=True,
    )
    
    if cfg.verbose:
        total_crops = sum(len(regions) for regions in picture_map.values())
        print(f"[INFO] Extracted {total_crops} picture regions across {len(picture_map)} pages")
        for img, regs in picture_map.items():
            print(f"  {img}: {len(regs)} picture(s)")

    if cfg.save_manifest:
        manifest_path = crops_dir.parent / cfg.manifest_filename
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        if cfg.verbose:
            print(f"[SAVE] Manifest: {manifest_path}")

    # 3) Gemma crop OCR + Summary generation
    if cfg.verbose:
        print(f"\n{'='*60}")
        print(f"STEP 3: Gemma Crop OCR & Summary Generation")
        print(f"{'='*60}")
    gemma_engine = GemmaCropOcrEngine(model_id=cfg.gemma_model_id, dtype=dtype, device_map=cfg.device_map)
    try:
        final_reports, gemma_timing = build_final_reports(
            manifest=manifest,
            ocr_map=parsed_map,
            gemma_engine=gemma_engine,
            crop_upscale=cfg.crop_upscale,
            image_folder=image_folder,
            summary_max_tokens=cfg.summary_max_tokens,
            verbose=cfg.verbose,
        )
    finally:
        if cfg.verbose:
            print(f"[GEMMA] Closing engine and clearing GPU memory...")
        gemma_engine.close()

    out_json_path = reports_dir / "final_image_description_reports_combined_batch_v2.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(final_reports, f, ensure_ascii=False, indent=2)

    if cfg.verbose:
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"[SAVE] Final report: {out_json_path}")
        print(f"[INFO] Processed {len(final_reports.get('pages', []))} pages")
        if final_reports.get('errors'):
            print(f"[WARN] {len(final_reports['errors'])} errors occurred")
        
        # Comprehensive timing summary
        print(f"\n{'='*60}")
        print(f"TIMING SUMMARY")
        print(f"{'='*60}")
        print(f"\nDOTS OCR:")
        print(f"  Total images     : {dots_timing['total_images']}")
        print(f"  Total time       : {dots_timing['total_time']:.2f}s")
        print(f"  Avg per image    : {dots_timing['avg_time_per_image']:.2f}s")
        print(f"  Generation time  : {dots_timing['total_gen_time']:.2f}s (avg {dots_timing['total_gen_time']/dots_timing['total_images']:.2f}s/image)")
        
        print(f"\nGEMMA OCR:")
        print(f"  Total pages      : {gemma_timing['total_pages']}")
        print(f"  Total crops      : {gemma_timing['total_crops']}")
        print(f"  Total time       : {gemma_timing['total_time']:.2f}s")
        print(f"  Avg per page     : {gemma_timing['avg_time_per_page']:.2f}s")
        print(f"  Crop OCR time    : {gemma_timing['total_crop_ocr_time']:.2f}s (avg {gemma_timing['avg_crop_ocr_time']:.2f}s/crop)")
        print(f"  Summary time     : {gemma_timing['total_summary_time']:.2f}s (avg {gemma_timing['avg_summary_time']:.2f}s/page)")
        
        total_pipeline_time = dots_timing['total_time'] + gemma_timing['total_time']
        print(f"\nOVERALL:")
        print(f"  Total pipeline   : {total_pipeline_time:.2f}s")
        print(f"  DOTS portion     : {dots_timing['total_time']:.2f}s ({dots_timing['total_time']/total_pipeline_time*100:.1f}%)")
        print(f"  Gemma portion    : {gemma_timing['total_time']:.2f}s ({gemma_timing['total_time']/total_pipeline_time*100:.1f}%)")
        print(f"{'='*60}")

    return out_json_path
