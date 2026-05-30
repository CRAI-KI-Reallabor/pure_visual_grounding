"""
Convert efficient_llm pipeline JSON output into RAG-ready chunks.

3 chunk types:
  - page_text  : all DOTS text/header/list/formula elements concatenated
  - table      : each table as its own chunk with HTML stripped to plaintext
  - picture    : each picture region (OCR text + adjacent text + description)
"""

import json
import re
import sys
from pathlib import Path


def strip_html_table(html: str) -> str:
    """Convert HTML table to readable key-value plaintext for embedding."""
    # Extract headers
    headers = re.findall(r"<th[^>]*>(.*?)</th>", html, re.DOTALL)
    headers = [re.sub(r"<[^>]+>", " ", h).strip() for h in headers]

    # Extract rows
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)

    lines = []
    col_headers = headers[1:] if len(headers) > 1 else []

    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
        cells = [re.sub(r"<[^>]+>", " ", c).strip() for c in cells]
        if not cells:
            continue
        row_label = cells[0]
        for i, val in enumerate(cells[1:]):
            if not val:
                continue
            col = col_headers[i] if i < len(col_headers) else f"Spalte {i+1}"
            if row_label:
                lines.append(f"{row_label} | {col}: {val}")
            else:
                lines.append(f"{col}: {val}")

    return "\n".join(lines) if lines else re.sub(r"<[^>]+>", " ", html).strip()


def make_page_text_chunk(page_image: str, pdf_name: str, ocr_elements: list) -> dict | None:
    """Build a chunk from all non-table, non-picture DOTS text elements."""
    text_categories = {"text", "section-header", "page-header", "page-footer",
                       "title", "list-item", "footnote", "formula", "caption"}
    parts = []
    for el in ocr_elements:
        cat = (el.get("category") or "").lower()
        if cat in text_categories and el.get("text"):
            parts.append(el["text"].strip())

    if not parts:
        return None

    return {
        "chunk_id": f"{pdf_name}__{page_image}__page_text",
        "chunk_type": "page_text",
        "source_pdf": pdf_name,
        "page": page_image,
        "content": "\n".join(parts),
    }


def make_table_chunks(page_image: str, pdf_name: str, ocr_elements: list) -> list:
    """Build one chunk per table on the page."""
    chunks = []
    table_idx = 0
    for el in ocr_elements:
        cat = (el.get("category") or "").lower()
        if cat == "table" and el.get("text"):
            plain = strip_html_table(el["text"])
            if not plain.strip():
                continue
            table_idx += 1
            chunks.append({
                "chunk_id": f"{pdf_name}__{page_image}__table_{table_idx}",
                "chunk_type": "table",
                "source_pdf": pdf_name,
                "page": page_image,
                "bbox": el.get("bbox"),
                "content": plain,
            })
    return chunks


def make_picture_chunks(page_image: str, pdf_name: str, picture_ocr_result: list) -> list:
    """Build one chunk per picture region combining OCR text + adjacent text + description."""
    chunks = []
    for region in picture_ocr_result:
        rid = region.get("region_id", "")
        parts = []

        ocr_text = (region.get("text") or "").strip()
        adjacent = (region.get("adjacent_text") or "").strip()
        description = (region.get("description") or "").strip()

        if ocr_text:
            parts.append(f"OCR Text:\n{ocr_text}")
        if adjacent:
            parts.append(f"Kontext:\n{adjacent}")
        if description:
            parts.append(f"Beschreibung:\n{description}")

        if not parts:
            continue

        chunks.append({
            "chunk_id": f"{pdf_name}__{page_image}__{rid}",
            "chunk_type": "picture",
            "source_pdf": pdf_name,
            "page": page_image,
            "region_id": rid,
            "bbox": region.get("bbox"),
            "content": "\n\n".join(parts),
        })
    return chunks


def create_rag_chunks(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pdf_name = data.get("pdf_name", "document")
    pages = data.get("pages", [])

    all_chunks = []

    for page in pages:
        page_image = page.get("page_image", "")
        ocr_result = page.get("OCR_Result", {})
        ocr_elements = ocr_result.get("ocr_pass_result", [])
        picture_ocr = ocr_result.get("picture_ocr_result", [])

        # Page text chunk
        text_chunk = make_page_text_chunk(page_image, pdf_name, ocr_elements)
        if text_chunk:
            all_chunks.append(text_chunk)

        # Table chunks
        all_chunks.extend(make_table_chunks(page_image, pdf_name, ocr_elements))

        # Picture chunks
        all_chunks.extend(make_picture_chunks(page_image, pdf_name, picture_ocr))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"total_chunks": len(all_chunks), "chunks": all_chunks}, f, ensure_ascii=False, indent=2)

    print(f"Created {len(all_chunks)} chunks -> {output_path}")

    # Print breakdown
    types = {}
    for c in all_chunks:
        t = c["chunk_type"]
        types[t] = types.get(t, 0) + 1
    for t, count in types.items():
        print(f"  {t}: {count}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_rag_chunks.py <input.json> <output_chunks.json>")
        sys.exit(1)
    create_rag_chunks(sys.argv[1], sys.argv[2])
