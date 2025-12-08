import os
import json
import time
from typing import Optional
from . import utils, prompts
from .engine import QwenEngine
from .config import LocalDualLLMConfig

def inference_pdf(pdf_path: str, config: LocalDualLLMConfig = None, engine: Optional[QwenEngine] = None, custom_output_dir: Optional[str] = None):
    # 1. Load Config if not provided
    if config is None:
        config = LocalDualLLMConfig()

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # 2. Setup Directories using CONFIG object
    current_dataset_dir = os.path.join(config.dataset_dir, pdf_name)
    final_output_dir = custom_output_dir if custom_output_dir else config.results_dir

    for d in [config.dataset_dir, config.cache_dir, config.debug_log_dir, current_dataset_dir, final_output_dir]:
        os.makedirs(d, exist_ok=True)

    # 3. PDF to Images
    image_paths = utils.convert_pdf_to_images(pdf_path, current_dataset_dir)
    if not image_paths:
        return {"status": "error", "message": "PDF Conversion failed"}

    # 4. Load Engine
    local_engine = False
    if engine is None:
        engine = QwenEngine(config=config)
        local_engine = True

    results_collection = {"pdf_name": pdf_name, "pages": [], "errors": []}

    try: 
        for image_path in image_paths:
            page_name = os.path.basename(image_path)
            try:
                # --- OCR Step ---
                ocr_msgs = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}", "min_pixels": config.min_pixels, "max_pixels": config.max_pixels},
                        {"type": "text", "text": prompts.OCR_PROMPT},
                    ]
                }]
                raw_ocr = engine.run_inference(ocr_msgs, config.gen_max_new_tokens_ocr)
                
                # Use the updated utils function
                ocr_result = utils.parse_ocr_json(raw_ocr, page_name)
                
                if ocr_result.get("ok"):
                    # Success
                    data = ocr_result["data"]
                    extracted_info = data.get("extracted_information", {})
                else:
                    # Failure
                    extracted_info = {}
                    log_path = os.path.join(config.debug_log_dir, f"ocr_fail_{page_name}.txt")
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(ocr_result.get("raw", ""))
                
                flattened_info = utils.flatten_extracted_info(extracted_info)

                # --- Report Step ---
                report_msgs = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}", "min_pixels": config.min_pixels, "max_pixels": config.max_pixels},
                        {"type": "text", "text": prompts.REPORT_PROMPT},
                        {"type": "text", "text": "OCR Reference:\n" + json.dumps(flattened_info, ensure_ascii=False)}
                    ]
                }]
                raw_report = engine.run_inference(report_msgs, config.gen_max_new_tokens_report)
                parsed_report = utils.parse_json_garbage(raw_report)

                results_collection["pages"].append({
                    "page_image": page_name,
                    "OCR_Result": flattened_info,
                    "Generated_Report": parsed_report
                })

            except Exception as e:
                results_collection["errors"].append({"page": page_name, "error": str(e)})
            
        # Save Results
        output_filename = f"{pdf_name}_Final_Report.txt"
        output_path = os.path.join(final_output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(results_collection, indent=2, ensure_ascii=False))
        
    finally:
        utils.cleanup_folder_using_os(current_dataset_dir)
        if local_engine:
            engine.close()

    return results_collection

def batched_inference(input_folder: str, config: LocalDualLLMConfig = None, engine: Optional[QwenEngine] = None):
    """
    Scans the input_folder for PDF files and processes them.
    
    Args:
        input_folder: Folder containing PDFs.
        config: Configuration object.
        engine: Optional pre-loaded engine. If None, one will be loaded temporarily.
    """
    start_time = time.time()
    
    # 1. Setup Config
    if config is None:
        config = LocalDualLLMConfig()

    # 2. Find PDF Files
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        return {"total": 0, "processed": [], "failed": []}

    files = [
        os.path.join(input_folder, f) 
        for f in os.listdir(input_folder) 
        if f.lower().endswith('.pdf')
    ]
    files.sort()
    
    if not files:
        print(f"No PDF files found in {input_folder}")
        return

    print(f"Found {len(files)} PDF files to process in batch.")

    # 3. Engine Logic
    local_engine_created = False
    if engine is None:
        print("Initializing Global Engine for Batch Processing...")
        engine = QwenEngine(config=config)
        local_engine_created = True

    summary = {"total": len(files), "processed": [], "failed": []}
    
    # 4. File Loop
    try:
        for idx, pdf_file in enumerate(files):
            print(f"\n[{idx+1}/{len(files)}] Starting: {os.path.basename(pdf_file)}")
            try:
                # Use shared engine
                inference_pdf(pdf_path=pdf_file, config=config, engine=engine)
                summary["processed"].append(pdf_file)
            except Exception as e:
                print(f"CRITICAL FAILURE on {pdf_file}: {e}")
                summary["failed"].append(pdf_file)
    finally:
        # 5. Cleanup only if we created engine for this run
        if local_engine_created:
            print("Cleaning up batch engine...")
            engine.close()
    # 6. Stats
    duration = time.time() - start_time
    print(f"\nBatch Complete. Time taken: {duration:.2f}s")
    print(f"Success: {len(summary['processed'])}, Failed: {len(summary['failed'])}")
    
    return summary


def recursive_batched_inference(input_root_dir: str, config: LocalDualLLMConfig = None, engine: Optional[QwenEngine] = None):
    """
    Recursively scans input_root_dir for PDFs.
    Recreates the directory structure in config.results_dir.
    """
    start_time = time.time()
    
    # 1. Setup Config
    if config is None:
        config = LocalDualLLMConfig()
        
    # Clean paths
    input_root_dir = os.path.normpath(input_root_dir)
    root_folder_name = os.path.basename(input_root_dir)
    
    print(f"=== Starting Recursive Batch Inference on: {input_root_dir} ===")
    
    # 2. Handle Engine Logic
    local_engine_created = False
    if engine is None:
        print("Initializing Engine for Recursive Batch...")
        engine = QwenEngine(config=config)
        local_engine_created = True
    
    stats = {"processed": 0, "failed": 0}

    try:
        # 3. Walk through the directory tree
        for current_root, dirs, files in os.walk(input_root_dir):
            # Filter for PDFs
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                continue
                
            # 4. Target Directory Structure
            relative_path = os.path.relpath(current_root, input_root_dir)
            
            if relative_path == ".":
                # Files in the root input directory
                target_subpath = ""
            else:
                # Handle subdirectories. 
                # Logic: Split path, append '_Final_Report' to every folder name
                path_parts = relative_path.split(os.sep)
                renamed_parts = [f"{p}_Final_Report" for p in path_parts]
                target_subpath = os.path.join(*renamed_parts)
                
            # Construct full target path: RESULTS_DIR / RootName / RelPath_Renamed
            target_dir = os.path.join(config.results_dir, root_folder_name, target_subpath)
            
            os.makedirs(target_dir, exist_ok=True)
            
            print(f"\n>> Entering Folder: {relative_path} -> Target: {target_dir}")
            
            # 5. Process Files
            for pdf_file in pdf_files:
                full_pdf_path = os.path.join(current_root, pdf_file)
                print(f"   Processing: {pdf_file}")
                
                try:
                    inference_pdf(
                        pdf_path=full_pdf_path, 
                        config=config,
                        engine=engine, 
                        custom_output_dir=target_dir
                    )
                    stats["processed"] += 1
                except Exception as e:
                    print(f"   FAILED: {pdf_file} - {e}")
                    stats["failed"] += 1
    finally:
        # 6. Cleanup
        if local_engine_created:
            engine.close()
    
    duration = time.time() - start_time
    print(f"\n=== Batch Complete ===\nTime: {duration:.2f}s\nProcessed: {stats['processed']}\nFailed: {stats['failed']}")
    return stats