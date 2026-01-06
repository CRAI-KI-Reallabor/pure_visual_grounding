# Pure Visual Grounding

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for processing PDFs with vision-based language models, specialized for technical document OCR and structured data extraction.

## Overview

Pure Visual Grounding leverages the power of vision-enabled large language models to extract structured information from PDF documents. This package is particularly useful for:

- **Technical Document Processing**: Extract structured data from complex technical documents
- **OCR with Context**: Go beyond simple OCR to understand document structure and meaning
- **RAG Pipeline Integration**: Prepare documents for Retrieval-Augmented Generation workflows
- **Automated Document Analysis**: Process large volumes of PDFs with consistent structured output

## Features

- 🔍 **Vision-based PDF Processing**: Uses advanced vision models for accurate document analysis
- 📄 **Multi-page Support**: Processes entire PDF documents page by page
- 🏗️ **Structured Output**: Returns organized JSON data with metadata for each page
- 🎯 **Customizable Prompts**: Configure extraction prompts for specific document types
- 🔄 **Two-pass Processing**: Initial extraction followed by refinement for improved accuracy
- 📊 **High-DPI Rendering**: Configurable DPI settings for optimal image quality
- 🛠️ **LangChain Integration**: Built on LangChain for easy model swapping and configuration

## Installation

Install from PyPI:

```bash
pip install pure-visual-grounding
```

## Quick Start

### Basic Usage

```python
from pure_visual_grounding import process_pdf_with_vision
from langchain_openai import ChatOpenAI

# Initialize your vision model
llm = ChatOpenAI(
    model="gpt-4-vision-preview",
    api_key="your-openai-api-key"
)

# Read PDF file as bytes
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

# Process the PDF
results = process_pdf_with_vision(
    pdf_name="document.pdf",
    pdf=pdf_bytes,
    llm=llm,
    vision_prompt="First prompt to get the information out of image",
    reinforced_prompt="Re inforced prompt to make sure all information is extracted"
)

# Access structured results
for page_result in results:
    print(f"Page {page_result['metadata']['page_number']}")
    print(f"Content: {page_result.get('content', 'No content extracted')}")
```

### Custom Processing

```python
from pure_visual_grounding import process_pdf_with_vision
from langchain_openai import ChatOpenAI

# Custom prompts for specific document types
custom_prompt = """
Analyze this technical document page and extract:
1. Section headings and hierarchy
2. Technical specifications and parameters
3. Diagrams, charts, and visual elements
4. Tables with numerical data
5. Key formulas or equations

Format the response as structured JSON with clear categorization.
"""

reinforcement_prompt = """
Review the previous analysis and enhance it by:
1. Ensuring all technical details are captured accurately
2. Organizing information hierarchically
3. Adding any missed visual elements
4. Validating numerical data and units
"""

llm = ChatOpenAI(model="gpt-4-vision-preview", api_key="your-key")

with open("technical_doc.pdf", "rb") as f:
    pdf_bytes = f.read()

results = process_pdf_with_vision(
    pdf_name="technical_doc.pdf",
    pdf=pdf_bytes,
    llm=llm,
    vision_prompt=custom_prompt,
    reinforced_prompt=reinforcement_prompt,
    dpi=300  # High resolution for detailed documents
)
```

## Advanced Usage

### Batch Processing

```python
import glob
from pathlib import Path

def process_pdf_batch(pdf_directory, llm):
    """Process multiple PDFs in a directory"""
    pdf_files = glob.glob(str(Path(pdf_directory) / "*.pdf"))
    results = {}

    for pdf_path in pdf_files:
        pdf_name = Path(pdf_path).name

        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            results[pdf_name] = process_pdf_with_vision(
                pdf_name=pdf_name,
                pdf=pdf_bytes,
                llm=llm,
                vision_prompt="First prompt to get the information out of image",
                reinforced_prompt="Re inforced prompt to make sure all information is extracted"
            )
            print(f"✓ Processed: {pdf_name}")

        except Exception as e:
            print(f"✗ Failed: {pdf_name} - {e}")
            results[pdf_name] = {"error": str(e)}

    return results

# Usage
llm = ChatOpenAI(model="gpt-4-vision-preview", api_key="your-key")
batch_results = process_pdf_batch("./documents/", llm)
```

### Integration with Different Models

```python
# OpenAI GPT-4 Vision
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4-vision-preview", api_key="your-key")

# Anthropic Claude (when vision is available)
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet-20240229", api_key="your-key")

# Google Gemini Vision
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", api_key="your-key")
```

## Output Format

The package returns a list of dictionaries, based on the provided output structure in prompt. For example,:

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

## Configuration Options

### Function Parameters

- **pdf_name** (str): Name identifier for the PDF file
- **pdf** (bytes): PDF content as bytes
- **llm** (BaseChatModel): LangChain vision-capable language model
- **vision_prompt** (str): Initial extraction prompt
- **reinforced_prompt** (str): Secondary refinement prompt
- **dpi** (int): Image resolution for PDF rendering (default: 300)

### Recommended DPI Settings

- **150 DPI**: Fast processing, basic documents
- **300 DPI**: Standard quality, most documents (default)
- **600 DPI**: High quality, detailed technical documents

## Use Cases

### Technical Documentation

```python
# Optimized for technical manuals, specifications, and research papers
tech_prompt = """
Extract from this technical document:
1. Technical specifications and parameters
2. Procedural steps and instructions
3. Diagrams, schematics, and their descriptions
4. Tables with measurements and data points
5. Safety warnings and important notes
Format as structured JSON with clear sections.
"""
```

### Financial Documents

```python
# Optimized for financial reports, statements, and analysis
finance_prompt = """
Analyze this financial document and extract:
1. Financial figures, ratios, and metrics
2. Tables with numerical data
3. Charts and graphs with their insights
4. Key financial statements sections
5. Important dates and periods
Return structured data suitable for financial analysis.
"""
```

### Research Papers

```python
# Optimized for academic and research publications
research_prompt = """
Process this research document and identify:
1. Abstract and key findings
2. Methodology and experimental setup
3. Results, data tables, and statistics
4. Figures, graphs, and their captions
5. References and citations
Structure the output for academic analysis.
"""
```

## Error Handling

The package includes robust error handling:

```python
try:
    results = process_pdf_with_vision(pdf_name, pdf_bytes, llm)

    for result in results:
        if result["metadata"]["error"] != "none":
            print(f"Page {result['metadata']['page_number']} had errors: {result['metadata']['error']}")
        else:
            # Process successful result
            print(f"Successfully processed page {result['metadata']['page_number']}")

except Exception as e:
    print(f"Fatal error processing PDF: {e}")
```

## Requirements

- Python 3.8+
- langchain ~= 0.3.27
- langchain-core ~= 0.3.72
- PyMuPDF (for PDF processing)
- pathlib ~= 1.0.1
- langsmith >= 0.1.17

## Performance Tips

1. **Optimize DPI**: Use 300 DPI for most documents; increase only for fine details
2. **Batch Processing**: Process multiple pages/documents in batches for efficiency
3. **Model Selection**: Choose appropriate models based on accuracy vs. speed requirements
4. **Prompt Engineering**: Craft specific prompts for your document types
5. **Error Handling**: Implement retry logic for transient API failures

## Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Author**: Strategion (development@strategion.de)

**Keywords**: PDF, OCR, Vision, LLM, Document Processing, Technical Documents, RAG, LangChain
