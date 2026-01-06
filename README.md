# Pure Visual Grounding (Meta Package)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

One package, multiple PDF-to-vision pipelines. The base install ships the original cloud/hosted-model flow; extras enable heavier local stacks such as `local_dual_llm` (Qwen2.5-VL).

## Approaches

- **Core (cloud/hosted vision models)** — LangChain-based, uses external vision LLMs.  
  - Install: `pip install pure-visual-grounding`
- **Local Dual LLM (Qwen2.5-VL)** — Fully local two-pass OCR + report pipeline.  
  - Install: `pip install pure-visual-grounding[local-dual-llm]`
- **Future techniques** — Add new flows as subpackages and expose via extras (see “Extendable pattern”).

## Feature Highlights

- 🔍 Vision-first PDF parsing with page-level structure
- 📄 Multi-page rendering and processing
- 🏗️ Structured JSON outputs tuned for technical docs
- 🔌 Pluggable techniques via extras, keeping base install light
- 🧰 Shared utilities for PDF rendering, JSON cleanup, and batching

## Install

- Base (core flow):  
  `pip install pure-visual-grounding`
- Local Qwen2.5-VL flow:  
  `pip install pure-visual-grounding[local-dual-llm]`

## Quick Start (Core Flow)

```python
from pure_visual_grounding import process_pdf_with_vision
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-vision-preview", api_key="your-key")

with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

results = process_pdf_with_vision(
    pdf_name="document.pdf",
    pdf=pdf_bytes,
    llm=llm,
    vision_prompt="First prompt to get the information out of image",
    reinforced_prompt="Re inforced prompt to make sure all information is extracted",
)
print(results[0]["content"])
```

## Quick Start (Local Dual LLM)

```python
from local_dual_llm import inference_pdf

result = inference_pdf("samples/document.pdf")
print(result["pages"][0]["Generated_Report"])
```

## When to Choose Which

- Use **Core** for quick setup with hosted vision models and minimal local deps.
- Use **Local Dual LLM** for offline/local runs, GPU acceleration, and controlled caching/prompts.

## Package Layout

- `pure_visual_grounding/` — Core LangChain-based vision pipeline (cloud/hosted)
- `local_dual_llm/` — Local Qwen2.5-VL pipeline (OCR + report)
- `examples/` (recommended) — Per-technique runnable samples
- `tests/` (recommended) — Technique-specific tests

## Output Shape (Core Flow Example)

```json
[
  {
    "content": "Extracted and structured content from the page",
    "metadata": {
      "pdf_name": "document.pdf",
      "page_number": 1,
      "error": "none",
      "processing_time": "2.34s",
      "model_used": "gpt-4-vision-preview"
    }
  }
]
```

## Performance Tips

- Pick model/device per technique; keep caches on fast storage.
- Batch pages/PDFs and reuse engines to avoid reload overhead.
- Tune DPI/pixel budgets (core) or token limits (local_dual_llm) for speed vs. recall.

## Contributing

- Keep public APIs stable; add new techniques via extras.
- Document new flows with a dedicated README and example.
- Prefer optional dependencies for heavy stacks.
- Extend tests and examples

## License

MIT License (see `LICENSE`).

---

**Author**: Strategion (development@strategion.de)

**Keywords**: PDF, OCR, Vision, LLM, Qwen2.5-VL, Document Processing, Technical Documents, RAG, LangChain

