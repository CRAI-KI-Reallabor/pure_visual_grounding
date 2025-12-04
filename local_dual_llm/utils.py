import os
import json
import re
import pymupdf as fitz  # PyMuPDF

def convert_pdf_to_images(pdf_path: str, output_folder: str) -> list:
    """
    Converts PDF to PNGs using PyMuPDF (no Poppler required),
    saves them, and returns list of file paths.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"Converting PDF: {pdf_path}...")
    
    saved_paths = []
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        for i, page in enumerate(doc):
            # Set Zoom for High Resolution (300 DPI)
            # Standard PDF is 72 DPI. 300 / 72 = ~4.166
            zoom = 300 / 72
            mat = fitz.Matrix(zoom, zoom)
            
            pix = page.get_pixmap(matrix=mat)
            
            file_name = f"{base_name}_page_{i+1}.png"
            full_path = os.path.join(output_folder, file_name)
            
            pix.save(full_path)
            saved_paths.append(full_path)
            
        doc.close()
        
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        return []
        
    return saved_paths

# JSON Tools
def parse_ocr_json(raw_text: any, image_name: str) -> dict:
    s = str(raw_text)
    try:
        # Try cleaning code blocks first
        json_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.DOTALL)
        data = json.loads(json_str)
        return {"ok": True, "data": data}
    except json.JSONDecodeError:
        try:
            # Fallback: finding the first outer braces
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1:
                data = json.loads(s[start:end+1])
                return {"ok": True, "data": data}
        except json.JSONDecodeError:
            pass
            
    # If we reach here, all parsing attempts failed
    print(f"[Warning] OCR for {image_name} failed to parse into JSON.")
    return {"ok": False, "raw": s}

def flatten_extracted_info(extracted_info):
    ei = dict(extracted_info or {})
    def _as_str(x): return str(x) if x not in (None, {}) else ""
    def _as_list(x): return x if isinstance(x, list) else [x] if x not in (None, "", {}) else []

    topic_src = dict(ei.get("Topic_and_context_information") or {})
    ti = topic_src.get("technical_identifier", ei.get("technical_identifier"))
    td = topic_src.get("topic_description", ei.get("topic_description"))
    ci = topic_src.get("context_information", ei.get("context_information"))
    
    ti, td, ci = _as_str(ti), _as_str(td), _as_str(ci)
    
    ei["Topic_and_context_information"] = {"technical_identifier": ti, "topic_description": td, "context_information": ci}
    ei["technical_identifier"] = ti
    ei["topic_description"] = td
    ei["context_information"] = ci
    
    for k in ["product_component_information", "embedded_table_chart", "side_margin_text", "product_measurement_information"]:
        ei[k] = _as_list(ei.get(k))
    return ei

def natural_sort_key(filename):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', filename)]

def parse_json_garbage(raw_text):
    """Parse JSON from raw text, handling markdown fences and malformed JSON."""
    if isinstance(raw_text, dict):
        return raw_text
    
    s = str(raw_text)
    try:
        # Try finding a JSON block first (with markdown fences)
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
        if match:
            return json.loads(match.group(1))
        # Fallback for non-fenced JSON
        start, end = s.find('{'), s.rfind('}')
        if start != -1 and end != -1:
            return json.loads(s[start:end+1])
    except json.JSONDecodeError:
        pass
    
    # If all parsing fails, return error dict
    return {"error": "Failed to parse JSON", "raw": s}

def parse_generated_report(report_str, image_name):
    """Simplified parser for the report."""
    if isinstance(report_str, dict):
        return report_str
    
    s = str(report_str)
    try:
        # Try finding a JSON block first
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
        if match:
            return json.loads(match.group(1))
        # Fallback for non-fenced JSON
        start, end = s.find('{'), s.rfind('}')
        if start != -1 and end != -1:
            return json.loads(s[start:end+1])
    except json.JSONDecodeError:
        print(f"[Warning] Report for {image_name} could not be parsed as JSON. Saving raw output.")
        # Create a fallback structure with the raw text
        return {
            "Core Theme Identification": {"core_topic": "Parsing Failed. See raw_output."},
            "Image_summary": {"Comprehensive Narrative": ""},
            "Missing_OCR_result": {"Missing_Product_information": []},
            "raw_output": s
        }
    # If all fails, return a basic fallback
    return {
        "Core Theme Identification": {"core_topic": "Parsing Failed. See raw_output."},
        "raw_output": s
    }

# Cleanup temporary dataset folder
def cleanup_folder_using_os(folder_path):
    """
    Recursively deletes a folder and its contents.
    Used to delete temp dataset folders created during runtime.
    """
    if not os.path.exists(folder_path):
        return

    # Walk the tree bottom-up (files first, then folders)
    for root, dirs, files in os.walk(folder_path, topdown=False):
        
        # 1. Delete all files in the current folder
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not delete file {file_name}: {e}")
        
        # 2. Delete all sub-folders (now empty)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                os.rmdir(dir_path)
            except Exception as e:
                print(f"Warning: Could not delete folder {dir_name}: {e}")

    # 3. Finally, delete the root folder itself
    try:
        os.rmdir(folder_path)
    except Exception as e:
        print(f"Warning: Could not delete root temp folder {folder_path}: {e}")