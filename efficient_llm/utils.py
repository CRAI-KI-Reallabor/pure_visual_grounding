import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import io

import pymupdf as fitz
from PIL import Image


# ---------------- PDF -> PNG ----------------

def convert_pdf_to_images(
    pdf_path: Union[str, Path],
    output_folder: Union[str, Path],
    dpi: int = 300,
    max_long_edge_px: Optional[int] = None,
    grayscale: bool = False,
) -> List[str]:
    """
    Render a PDF to PNGs with configurable DPI and optional downscaling.

    Args:
        pdf_path: Input PDF path.
        output_folder: Directory to write PNGs.
        dpi: Target DPI for rasterization (default 300).
        max_long_edge_px: If set, downscale images so longest edge <= this value.
        grayscale: If True, render in grayscale to reduce size.
    """
    pdf_path = str(pdf_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            colorspace = fitz.csGRAY if grayscale else None
            pix = page.get_pixmap(matrix=mat, colorspace=colorspace)

            # Convert pixmap to PIL for optional downscale
            img_bytes = pix.tobytes("png")
            im = Image.open(io.BytesIO(img_bytes))

            if max_long_edge_px:
                im.thumbnail((max_long_edge_px, max_long_edge_px), Image.LANCZOS)

            # Convert to RGB if not grayscale to ensure consistency
            if not grayscale and im.mode not in ('RGB', 'L'):
                im = im.convert('RGB')

            file_name = f"{base_name}_page_{i+1}.png"
            full_path = output_folder / file_name
            # Use moderate compression for balance between size and speed
            im.save(full_path, format="PNG", optimize=True, compress_level=6)
            saved_paths.append(str(full_path))
    finally:
        doc.close()

    return saved_paths


# ---------------- JSON parsing helpers ----------------

def extract_json_array(raw_text: str) -> List[Dict[str, Any]]:
    raw = (raw_text or "").strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []

def _is_filename_like(k: str) -> bool:
    return isinstance(k, str) and any(k.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"])

def _try_json_load_maybe_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v

def parse_ocr_file(path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(data, dict) and any(_is_filename_like(k) for k in data.keys()):
        for img_name, val in data.items():
            val2 = _try_json_load_maybe_list(val)
            out[img_name] = val2 if isinstance(val2, list) else []
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("image_name") or item.get("filename") or item.get("file") or ""
            elems = None
            if "ocr" in item and isinstance(item["ocr"], list):
                elems = item["ocr"]
            elif "ocr_elements" in item and isinstance(item["ocr_elements"], list):
                elems = item["ocr_elements"]
            elif "extracted_information" in item and isinstance(item["extracted_information"], dict):
                ei = item["extracted_information"]
                if isinstance(ei.get("ocr_elements"), list):
                    elems = ei["ocr_elements"]
                elif isinstance(ei.get("layout"), list):
                    elems = ei["layout"]
            if name and isinstance(elems, list):
                out[name] = elems
    return out


# ---------------- Cropping helpers ----------------

def extract_picture_regions(ocr_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    for img, elems in ocr_map.items():
        pics: List[Dict[str, Any]] = []
        for idx, el in enumerate(elems):
            cat = (el.get("category") or el.get("type") or "")
            if isinstance(cat, str) and cat.lower() == "picture":
                bbox = el.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(n, (int, float)) for n in bbox):
                    pics.append({
                        "region_id": f"{Path(img).stem}_pic_{idx:04d}",
                        "bbox": bbox,
                        "category": "Picture",
                        "order": idx
                    })
        results[img] = pics
    return results

def pad_and_clip_bbox(bbox: List[float], padding: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(int, bbox)
    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(W, x2 + padding)
    y2p = min(H, y2 + padding)
    if x2p <= x1p: x2p = min(W, x1p + 1)
    if y2p <= y1p: y2p = min(H, y1p + 1)
    return x1p, y1p, x2p, y2p

def crop_image(page_image_path: Union[str, Path], bbox: List[float], padding: int, out_path: Union[str, Path]) -> bool:
    try:
        im = Image.open(page_image_path).convert("RGB")
    except Exception:
        return False
    W, H = im.size
    x1, y1, x2, y2 = pad_and_clip_bbox(bbox, padding, W, H)
    crop = im.crop((x1, y1, x2, y2))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return True

def build_picture_manifest(
    picture_map: Dict[str, List[Dict[str, Any]]],
    image_root: Union[str, Path],
    crops_dir: Union[str, Path],
    padding: int = 12,
    do_crop: bool = True
) -> Dict[str, Any]:
    manifest = {"images": []}
    image_root = Path(image_root)
    crops_dir = Path(crops_dir)

    for img, regions in picture_map.items():
        page_path = image_root / img
        page_entry = {
            "image_name": img,
            "page_image_path": str(page_path),
            "padding_px": padding,
            "picture_regions": []
        }
        W = H = None
        try:
            with Image.open(page_path) as im:
                W, H = im.size
        except Exception:
            pass

        for r in regions:
            rid = r["region_id"]
            bbox = r["bbox"]
            order = r.get("order")
            out_name = f"{rid}.png"
            crop_path = crops_dir / out_name
            saved = False
            if do_crop and page_path.exists():
                saved = crop_image(page_path, bbox, padding, crop_path)
            page_entry["picture_regions"].append({
                "region_id": rid,
                "bbox": bbox,
                "crop_image_path": str(crop_path) if (do_crop and saved) else None,
                "order": order,
                "width_px": W,
                "height_px": H
            })
        manifest["images"].append(page_entry)
    return manifest

def upscale_image(img: Image.Image, scale: float = 2.0) -> Image.Image:
    W, H = img.size
    return img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
