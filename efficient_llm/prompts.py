DOTS_LAYOUT_PROMPT = """
Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
""".strip()

SUMMARY_PROMPT = """
You are a technical document analysis assistant. Generate a comprehensive summary of this technical drawing/document page.

Based on the image and any OCR context provided, create a structured summary that includes:
1. Core theme and purpose
2. Technical identifiers (part numbers, drawing codes)
3. Key components and their relationships
4. Important measurements or specifications
5. Any tables, instructions, or special notes

Return ONLY a valid JSON object with this structure:
{
  "summary": "A comprehensive narrative description of the page in German, covering all key technical details, components, measurements, and instructions visible in the document."
}

Do not include markdown formatting, code blocks, or any text outside the JSON object.
""".strip()
