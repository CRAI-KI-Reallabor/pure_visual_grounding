import base64
import logging
from typing import Iterable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from .utils import clean_metadata
import pymupdf

from .json_cleaner import parse_generated_report
from .prompts import VISION_JSON_STRUCTURED_PROMPT, VISION_JSON_STRUCTURED_PROMPT_REINFORCED


def process_pdf_with_vision(pdf_name: str,
                            pdf: bytes,
                            llm: BaseChatModel,
                            vision_prompt: str,
                            reinforced_prompt: str,
                            dpi: int = 300):
    """
    Process PDF with vision-based language model for OCR and technical document analysis.
    
    Args:
        pdf_name (str): Name of the PDF file
        pdf (bytes): PDF file as bytes
        llm (BaseChatModel): LangChain language model with vision capabilities
        vision_prompt (str): Initial vision prompt for OCR extraction
        reinforced_prompt (str): Reinforcement prompt for improved results
        dpi (int): DPI for image rendering (default: 300)
    
    Returns:
        list: List of image description results with metadata for each page
    """
    image_description_results = []

    try:
        doc: pymupdf.Document = pymupdf.open(stream=pdf, filetype="pdf")

    except Exception:
        logging.error("Error reading the pdf file")

    pages: Iterable = doc.pages()

    for i, page in enumerate(pages):
        page_number = i
        
        logging.info(f"Processing Page {page_number + 1}")

        try:
            pix = page.get_pixmap(dpi=dpi)
            image_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{base64_image}"

            # First pass
            first_response = llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ]
                    )
                ]
            )

            # Second pass
            response = llm.invoke(
                [
                    SystemMessage(content=reinforced_prompt),
                    HumanMessage(
                        content=[
                            {"type": "text", "text": first_response.content},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ]
                    ),
                ]
            )

            result = parse_generated_report(response.content)

            metadata = {
                "pdf_name": pdf_name,
                "page_number": page_number + 1,
                "error": "none",
            }

            if("metadata" in result.keys()):
                metadata.update(result["metadata"])
            
            metadata = clean_metadata(metadata)

            result["metadata"] = metadata
            image_description_results.append(result)

            logging.info("File processing Successful.")

        except Exception as e:
            logging.error(f" ERROR Page {page_number}: {e}")
            metadata = {"pdf_name": pdf_name, "page_number": page_number + 1, "error": str(e)}
            image_description_results.append({"metadata": metadata})

    doc.close()

    return image_description_results