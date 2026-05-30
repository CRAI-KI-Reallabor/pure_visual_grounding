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

PICTURE_DESCRIPTION_PROMPT = """
You are an expert technical drawing analyst. Your task is to semantically describe the engineering drawing or diagram shown in this cropped image.

Describe:
1. What the diagram depicts (assembly, cross-section, schematic, flow chart, exploded view, etc.)
2. Key components, parts, and their spatial relationships
3. Any measurements, tolerances, or specifications visible
4. Labels, annotations, arrows, and what they indicate
5. The functional purpose of the depicted element if inferrable

Return ONLY a valid JSON object with this structure:
{
  "description": "Eine detaillierte semantische Beschreibung der Zeichnung auf Deutsch, einschließlich Komponenten, Beziehungen, Maße und Zweck."
}

Do not include markdown formatting, code blocks, or any text outside the JSON object.
""".strip()

SUMMARY_PROMPT = """
You are a technical document analysis assistant specializing in manufacturing and assembly documentation.

You will receive:
- A full page image of a technical document
- OCR context containing extracted text from the page and, if available, semantic descriptions of each engineering drawing on the page under "picture_descriptions"

Generate a comprehensive German-language summary that:
1. States the core theme and purpose of the page
2. Explicitly references all part numbers, drawing codes, and technical identifiers
3. Describes each diagram or drawing on the page by name and function, using the provided picture_descriptions if available
4. Includes all measurements, tolerances, and specifications in their correct context
5. Summarizes assembly steps, instructions, or procedures in order
6. Mentions any warnings, notes, or special conditions

Important: If picture_descriptions are provided in the context, you MUST incorporate their content into the summary. Do not ignore the diagram descriptions.

Return ONLY a valid JSON object with this structure:
{
  "summary": "Umfassende deutsche Zusammenfassung der Seite, die alle technischen Details, Teilenummern, Diagrammbeschreibungen, Maße und Montageschritte abdeckt."
}

Do not include markdown formatting, code blocks, or any text outside the JSON object.
""".strip()
