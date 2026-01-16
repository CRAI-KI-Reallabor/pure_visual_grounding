# Advanced Configuration and Technical Details

This document contains detailed technical information, advanced configuration options, and performance benchmarks for the `efficient_llm` module.

---

## Table of Contents

1. [Recommended CUDA / PyTorch / Flash-Attention Setup](#recommended-cuda--pytorch--flash-attention-setup)
2. [Transformer Version Compatibility](#transformer-version-compatibility)
3. [Advanced Programmatic Usage](#advanced-programmatic-usage)
4. [Configuration Reference](#configuration-reference)
5. [Performance Experiments](#performance-experiments)
6. [Memory Optimization Strategies](#memory-optimization-strategies)
7. [Batch Processing Guidelines](#batch-processing-guidelines)

---

## Recommended CUDA / PyTorch / Flash-Attention Setup

Below is a **tested working setup** verified on multiple systems.

### Tested Configuration

**Production Environment:**
- **OS**: Ubuntu Linux
- **Python**: 3.12.12
- **CUDA Toolkit**: 13.0 V13.0.88
- **PyTorch**: 2.7.0+cu126
- **Transformers**: 4.57.3
- **Flash-Attention**: 2.8.0.post2
- **GPU**: NVIDIA RTX 5000 Ada Generation (32GB VRAM)
- **Pillow**: 11.3.0
- **PyMuPDF**: 1.26.7

### 0. Install CUDA Toolkit (Required)

**Flash-Attention requires CUDA 11.8 or later.**

#### Download CUDA

Official CUDA Toolkit downloads:
- **CUDA 13.x (Latest)**: https://developer.nvidia.com/cuda-downloads
- **CUDA 12.x**: https://developer.nvidia.com/cuda-downloads
- **CUDA 11.8**: https://developer.nvidia.com/cuda-11-8-0-download-archive

**Tested versions**: CUDA 12.1, CUDA 13.0  
**Minimum version**: CUDA 11.8

Select your operating system and follow the installation instructions.

#### Verify CUDA Installation

```bash
nvcc --version
```

Expected output for CUDA 13.0:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```

Expected output for CUDA 12.1:
```
Cuda compilation tools, release 12.1, Vxx.x.xxx
```

Check NVIDIA driver:
```bash
nvidia-smi
```

You should see your GPU listed with driver version and CUDA version.

### 1. Create Conda environment

```bash
conda create -n rednote_ocr python=3.12 -y
conda activate rednote_ocr
```

### 2. Install CUDA-enabled PyTorch

**Important**: PyTorch CUDA version should be compatible with your installed CUDA toolkit.

For CUDA 13.0 (use CUDA 12.4 or 12.6 PyTorch builds, which are forward-compatible):

```bash
# CUDA 12.6 wheel (tested with CUDA 13.0)
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126

# Or CUDA 12.4 wheel (also compatible)
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

For CUDA 12.1:

```bash
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Note**: PyTorch doesn't have separate CUDA 13.x wheels yet. Use CUDA 12.4 or 12.6 wheels for CUDA 13.0 - they are forward-compatible.

Or check PyTorch website for your specific configuration:
- https://pytorch.org/get-started/locally/

#### Verify PyTorch CUDA Installation

```python
import torch

print("PyTorch version:", torch.__version__)        # e.g. 2.7.0+cu126
print("CUDA version:", torch.version.cuda)         # e.g. 12.6
print("CUDA available:", torch.cuda.is_available()) # Should be True
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("GPU count:", torch.cuda.device_count())
```

Expected output:
```
PyTorch version: 2.7.0+cu126
CUDA version: 12.6
CUDA available: True
GPU: NVIDIA RTX 5000 Ada Generation
GPU count: 1
```

### 3. Install Flash-Attention (Required)

**Flash-Attention is required** for this pipeline and must be compiled with CUDA support.

**Official Resources:**
- **GitHub Repository**: https://github.com/Dao-AILab/flash-attention
- **Installation Guide**: https://github.com/Dao-AILab/flash-attention#installation-and-features
- **Documentation**: https://github.com/Dao-AILab/flash-attention/tree/main/docs

**Important**: Review the official installation guide before proceeding, especially for troubleshooting build errors or platform-specific requirements.

Install Flash-Attention:

```bash
pip install "flash-attn==2.8.0.post2" --no-build-isolation
```

**Installation Requirements**: 
- You **must** use `--no-build-isolation` for correct installation
- Requires CUDA toolkit installed and accessible
- Requires compatible C++ compiler (gcc/g++ on Linux, MSVC on Windows)
- This will compile Flash-Attention with your installed CUDA toolkit
- **Warning**: Compilation can take significant time (5-10 minutes typical, up to 2 hours on some systems depending on CPU and available cores)

**Tested Version**: `2.8.0.post2` is the verified version for this pipeline. Other versions (e.g. `2.8.2`) may work but are not tested.

#### Verify Flash-Attention Installation

Run this comprehensive check:

```python
import torch
import flash_attn

print("="*60)
print("System Check")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"\nFlash-Attention version: {flash_attn.__version__}")
print(f"Flash-Attention CUDA support: {hasattr(flash_attn, 'flash_attn_func')}")

# Test Flash-Attention on GPU
if torch.cuda.is_available():
    try:
        from flash_attn import flash_attn_func
        print("\nFlash-Attention function imported successfully")
        print("✓ Flash-Attention is properly installed and working with CUDA")
    except ImportError as e:
        print(f"\n✗ Error importing Flash-Attention: {e}")
else:
    print("\n✗ WARNING: CUDA not available. Flash-Attention requires GPU.")
```

Expected output:
```
============================================================
System Check
============================================================
PyTorch version: 2.7.0+cu126
CUDA available: True
CUDA version: 12.6
GPU: NVIDIA RTX 5000 Ada Generation

Flash-Attention version: 2.8.0.post2
Flash-Attention CUDA support: True

Flash-Attention function imported successfully
✓ Flash-Attention is properly installed and working with CUDA
```

---

## Transformer Version Compatibility

### The Problem

- Legacy DOTS OCR code was tied to older `transformers` versions
- Newer Gemma models require modern `transformers`
- After upgrading `transformers`, DOTS failed because its custom processor class no longer matched the expected signature

Recent `transformers` versions expect a `video_processor` argument in `Qwen2_5_VLProcessor`, which was missing in the original DOTS processor wrapper.

### The Fix

The DOTS processor implementation was updated so DOTS and Gemma can share the **same modern `transformers` version**.

**Modified file** (inside DOTS model repo):

```text
dots.ocr/weights/DotsOCR/configuration_dots.py
```

**Updated processor implementation**:

```python
class DotsVLProcessor(Qwen2_5_VLProcessor):
    attributes = ["image_processor", "tokenizer", "video_processor"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor,
            tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )

        self.image_token = "<|imgpad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.image_token_id = 151665 if not hasattr(tokenizer, "image_token_id") else tokenizer.image_token_id
```

The processor is then registered:

```python
AutoProcessor.register("dots_ocr", DotsVLProcessor)
CONFIG_MAPPING.register("dots_ocr", DotsOCRConfig)
```

### Result

- DOTS OCR works with the latest `transformers` versions
- Gemma runs in the same environment without downgrading dependencies
- No changes required in the `efficient_llm` pipeline code

The `pvg-download-ocr` CLI tool automatically applies this patch.

---

## Advanced Programmatic Usage

### Batch Multiple PDFs with Shared Engines

For maximum throughput, you can create shared DOTS and Gemma engines and reuse them across runs:

```python
import json
from pathlib import Path
from efficient_llm.config import PipelineConfig
from efficient_llm.engine import DotsLayoutEngine, GemmaCropOcrEngine
from efficient_llm.pipeline import run_dots_layout_on_folder, build_final_reports
from efficient_llm.utils import convert_pdf_to_images, extract_picture_regions, build_picture_manifest
from efficient_llm.prompts import DOTS_LAYOUT_PROMPT
import torch

# Initialize engines once
dtype = torch.bfloat16
dots_engine = DotsLayoutEngine(
    model_path="/path/to/DotsOCR",
    attn_impl="flash_attention_2",
    torch_dtype=dtype,
    device_map="auto",
)
gemma_engine = GemmaCropOcrEngine(
    model_id="google/gemma-3n-e4b-it",
    dtype=dtype,
    device_map="auto",
)

pdfs = sorted(Path("./pdfs").glob("*.pdf"))

try:
    for pdf in pdfs:
        print(f"\nProcessing: {pdf.name}")
        
        # Setup directories
        image_folder = Path(f"./batch/{pdf.stem}/pngs")
        crops_dir = Path(f"./batch/{pdf.stem}/crops")
        reports_dir = Path(f"./batch/{pdf.stem}/reports")
        
        image_folder.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert PDF
        convert_pdf_to_images(pdf, image_folder, dpi=200)
        
        # DOTS OCR
        ocr_map = run_dots_layout_on_folder(
            engine=dots_engine,
            image_folder=image_folder,
            prompt=DOTS_LAYOUT_PROMPT,
            batch_size=18,
            base_max_new_tokens=4000,
            verbose=True,
        )
        
        # Extract and crop picture regions
        picture_map = extract_picture_regions(ocr_map)
        manifest = build_picture_manifest(
            picture_map=picture_map,
            image_root=image_folder,
            crops_dir=crops_dir,
            padding=12,
            do_crop=True,
        )
        
        # Gemma OCR on crops
        final_reports = build_final_reports(
            manifest=manifest,
            ocr_map=ocr_map,
            gemma_engine=gemma_engine,
            crop_upscale=2.0,
            image_folder=image_folder,
            verbose=True,
        )
        
        # Save report
        out_path = reports_dir / "final_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_reports, f, ensure_ascii=False, indent=2)
        
        print(f"Completed: {pdf.name} -> {out_path}")

finally:
    dots_engine.close()
    gemma_engine.close()
```

### Custom Processing with Fine-Grained Control

```python
from efficient_llm.config import PipelineConfig
from efficient_llm.pipeline import run_pipeline
from pathlib import Path

# Process with maximum quality settings
cfg = PipelineConfig(
    dots_model_path="/path/to/DotsOCR",
    pdf_path="technical_drawing.pdf",
    image_folder="./high_quality/pngs",
    crops_dir="./high_quality/crops",
    reports_dir="./high_quality/reports",
    
    # High-quality PDF rendering
    pdf_dpi=300,
    max_long_edge_px=None,  # No downscaling
    
    # Conservative batch size for stability
    batch_size=6,
    
    # High token budgets for complex layouts
    base_max_new_tokens=8000,
    extra_max_new_tokens=12000,
    
    # Maximum crop upscaling for fine details
    crop_upscale=3.0,
    
    # Larger padding for context
    padding=20,
    
    # Flash attention for efficiency
    attn_impl="flash_attention_2",
    
    # Precision settings
    torch_dtype="bfloat16",
    device_map="auto",
    
    # Enable all logging
    verbose=True,
)

out_path = run_pipeline(cfg)
print(f"High-quality report: {out_path}")
```

---

## Configuration Reference

### PipelineConfig Parameters

#### Model Paths
- **`dots_model_path`** (str): Path to DOTS model directory **(required)**
- **`gemma_model_id`** (str): Hugging Face model ID for Gemma (default: `"google/gemma-3n-e4b-it"`)

#### I/O Paths
- **`pdf_path`** (str, optional): Input PDF file path
- **`image_folder`** (str): Directory for PNG pages **(required)**
- **`crops_dir`** (str): Directory for cropped picture regions **(required)**
- **`reports_dir`** (str): Directory for final JSON reports **(required)**

#### PDF Rendering
- **`pdf_dpi`** (int): DPI for PDF → PNG conversion (default: `200`)
- **`max_long_edge_px`** (int, optional): Maximum pixel size for longest edge; enables downscaling if set

#### DOTS Generation
- **`batch_size`** (int): Number of images processed simultaneously (default: `18`)
- **`base_max_new_tokens`** (int): Initial token budget for DOTS generation (default: `12000`)
- **`extra_max_new_tokens`** (int): Token budget for retry if JSON parsing fails (default: `12000`)
- **`attn_impl`** (str): Attention implementation: `"flash_attention_2"`, `"sdpa"`, or `"eager"` (default: `"flash_attention_2"`)

#### Picture Region Processing
- **`padding`** (int): Padding in pixels around detected picture bounding boxes (default: `12`)
- **`crop_upscale`** (float): Upscale factor for crops before Gemma OCR (default: `2.0`)

#### Device & Precision
- **`device_map`** (str): Device placement strategy: `"auto"`, `"cuda"`, `"cpu"` (default: `"auto"`)
- **`torch_dtype`** (str): Tensor dtype: `"bfloat16"`, `"float16"`, `"float32"` (default: `"bfloat16"`)

#### Output Control
- **`save_manifest`** (bool): Save picture regions manifest JSON (default: `False`)
- **`manifest_filename`** (str): Manifest filename (default: `"picture_regions_manifest_summary.json"`)
- **`verbose`** (bool): Enable detailed logging (default: `False`)

---

## Performance Experiments

### Hardware Environment

All measurements were taken on:

- **CPU:** AMD Ryzen Threadripper PRO 5955WX (16C/32T)
- **RAM:** 64 GB
- **GPU:** NVIDIA RTX 5000 Ada Generation  
  - Architecture: Ada Lovelace  
  - VRAM: 32 GB GDDR6  
  - Tensor Cores: Yes (bf16 / fp16 supported)

### Baseline: No Flash-Attention, No Real Batching

Initial implementation:
- Per-page processing (batch size ≈ 1)
- No Flash-Attention enabled
- High token budgets causing frequent retries

**Results:**
- **Inference time:** ~25-35s per image
- **VRAM usage:** ~20-26 GB
- **Main bottleneck:** DOTS generation + JSON retry logic

### Intermediate: Flash-Attention Without Batching

Enabled Flash-Attention but kept single-page processing:

**Results:**
- **VRAM usage:** Still ~26 GB
- **Throughput:** Modest improvements only
- **Conclusion:** Flash-Attention alone insufficient without batching

### Final: Flash-Attention + True Batched Inference

Restructured pipeline for true batched generation with Flash-Attention:

| Batch Size | `max_new_tokens` | VRAM (GB) | Time/Image (s) | Notes |
|----------:|------------------|----------:|---------------:|-------|
| 2 | 12000 | ~6 | ~34 | Small batch, first test |
| 6 | 12000 | ~8 | ~15 | Clear improvement |
| 9 | 12000 | ~8 | ~11 | Good balance |
| 18 | 12000 | ~13 | ~10 | High utilization |
| 18 | 4000 | ~13.5 | **~10-11** | **DOTS optimal: eliminates retries** |

**Note**: These times are for DOTS OCR only. Total pipeline time including Gemma summaries is ~50-60s per page.

### Best Configuration

**Recommended for RTX 5000 Ada (32GB VRAM):**

```python
cfg = PipelineConfig(
    batch_size=18,
    base_max_new_tokens=4000,
    attn_impl="flash_attention_2",
    torch_dtype="bfloat16",
    pdf_dpi=200,
)
```

**Performance:**
- **DOTS speed:** ~10-11 seconds per page
- **Gemma crop OCR:** ~4-5 seconds per crop (depends on number of picture regions)
- **Gemma summaries:** ~30-40 seconds per page (slowest component)
- **Overall pipeline:** ~50-60 seconds per page total
- **VRAM:** ~13-14 GB
- **Speedup vs baseline:** 2-3× faster per page (DOTS component is 4-5× faster)

**Performance Breakdown:**
- DOTS OCR: ~18-20% of total pipeline time (batched)
- Gemma crop OCR: ~20-25% of total time
- Gemma summaries: ~60-65% of total time (per-page, not batched)

---

## Memory Optimization Strategies

### Understanding Pipeline Performance

The pipeline has three main components with different performance characteristics:

1. **DOTS OCR** (~18-20% of time)
   - Batched processing (18 pages at once)
   - Flash-Attention enabled
   - Fast: ~10-11s per page

2. **Gemma Crop OCR** (~20-25% of time)
   - Per-crop processing
   - Small images
   - Fast: ~4-5s per crop

3. **Gemma Page Summaries** (~60-65% of time) ⚠️ **Performance Bottleneck**
   - Per-page processing (not batched)
   - Full-resolution images
   - Long context (all OCR results)
   - 2048 token generation
   - Slow: ~30-40s per page

**Key Insight**: Gemma summaries dominate total pipeline time. If you don't need page summaries, consider making them optional for 60-65% speedup.

### Reducing VRAM Usage

1. **Lower batch size:**
   ```python
   cfg = PipelineConfig(batch_size=6)  # or 9
   ```

2. **Reduce token budgets:**
   ```python
   cfg = PipelineConfig(
       base_max_new_tokens=4000,
       extra_max_new_tokens=6000,
   )
   ```

3. **Downscale input images:**
   ```python
   cfg = PipelineConfig(
       pdf_dpi=150,
       max_long_edge_px=1536,
   )
   ```

4. **Use float16 instead of bfloat16:**
   ```python
   cfg = PipelineConfig(torch_dtype="float16")
   ```

5. **Reduce crop upscaling:**
   ```python
   cfg = PipelineConfig(crop_upscale=1.5)
   ```

### Memory Budget Guidelines

| GPU VRAM | Recommended Batch Size | DPI | Expected Usage |
|----------|------------------------|-----|----------------|
| 8 GB | 2-3 | 150 | ~6-7 GB |
| 12 GB | 4-6 | 200 | ~8-10 GB |
| 16 GB | 6-9 | 200 | ~10-12 GB |
| 24 GB | 12-15 | 250 | ~14-18 GB |
| 32 GB | 18+ | 300 | ~13-20 GB |

---

## Batch Processing Guidelines

### Sequential Processing (Simple)

```python
from pathlib import Path
from efficient_llm.config import PipelineConfig
from efficient_llm.pipeline import run_pipeline

pdfs = list(Path("./input").glob("*.pdf"))

for pdf in pdfs:
    cfg = PipelineConfig(
        dots_model_path="/path/to/DotsOCR",
        pdf_path=str(pdf),
        image_folder=f"./output/{pdf.stem}/pngs",
        crops_dir=f"./output/{pdf.stem}/crops",
        reports_dir=f"./output/{pdf.stem}/reports",
    )
    
    try:
        out_path = run_pipeline(cfg)
        print(f"Success: {pdf.name}")
    except Exception as e:
        print(f"Failed: {pdf.name} - {e}")
```

### Parallel Processing (Advanced)

For truly parallel processing across multiple PDFs, consider:

1. **Multi-GPU setup:** Assign different PDFs to different GPUs
2. **Process pooling:** Use Python multiprocessing with GPU assignment
3. **Queue-based system:** Implement a job queue with worker processes

**Note:** Each worker should create its own engine instances to avoid CUDA context conflicts.

---

## CLI Advanced Examples

### Maximum Quality Mode

```bash
python -m efficient_llm.run_pipeline \
  --dots-model "/path/to/DotsOCR" \
  --pdf "technical_drawing.pdf" \
  --pdf-dpi 300 \
  --crop-upscale 3.0 \
  --padding 20 \
  --base-max-new-tokens 8000 \
  --batch-size 6
```

### Memory-Constrained Mode (8GB GPU)

```bash
python -m efficient_llm.run_pipeline \
  --dots-model "/path/to/DotsOCR" \
  --pdf "document.pdf" \
  --pdf-dpi 150 \
  --max-long-edge-px 1280 \
  --batch-size 3 \
  --base-max-new-tokens 4000 \
  --crop-upscale 1.5
```

### Speed-Optimized Mode

```bash
python -m efficient_llm.run_pipeline \
  --dots-model "/path/to/DotsOCR" \
  --pdf "document.pdf" \
  --pdf-dpi 200 \
  --batch-size 18 \
  --base-max-new-tokens 4000 \
  --attn-impl flash_attention_2
```

---

## Debug and Logging

### Enable Verbose Output

```python
cfg = PipelineConfig(
    verbose=True,
    # ... other config
)
```

### Check CUDA Status

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))
print("Memory allocated:", torch.cuda.memory_allocated(0) / 1e9, "GB")
print("Memory reserved:", torch.cuda.memory_reserved(0) / 1e9, "GB")
```

### Monitor GPU Usage During Execution

```bash
watch -n 1 nvidia-smi
```

---

## Troubleshooting Advanced Issues

### Flash-Attention Build Failures

If `flash-attn` installation fails:

1. Ensure CUDA Toolkit is installed and matches PyTorch CUDA version
2. Install build dependencies:
   ```bash
   pip install packaging wheel setuptools
   ```
3. Try without isolation:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

### Transformer Configuration Errors

If you see `KeyError` or attribute errors related to `video_processor`:

1. Ensure you're using `pvg-download-ocr` to download DOTS with patches
2. Or manually apply the processor patch (see Transformer Compatibility section)

### OOM Despite Low Batch Size

If OOM occurs even with batch_size=1:

1. Reduce image resolution drastically:
   ```bash
   --pdf-dpi 100 --max-long-edge-px 1024
   ```
2. Switch to float16:
   ```bash
   --torch-dtype float16
   ```
3. Close other GPU-using processes
4. Check for memory leaks (restart Python kernel between runs)

---

## Contributing

When contributing code:

1. Keep prompts in `prompts.py` strictly JSON-focused
2. Maintain explicit GPU cleanup in engine classes
3. Add verbose logging for new operations
4. Update this documentation for new features

---

## Keywords

PDF, OCR, Vision, Qwen2.5-VL, Document Processing, Technical Documents, DOTS OCR, Gemma, Vision-Language Models, Layout-aware OCR, Technical Drawings, Engineering Documents, Crop OCR, PDF-to-Image, Batched Inference, FlashAttention, GPU Inference, Local-first Inference, Structured JSON Outputs, Performance Optimization
