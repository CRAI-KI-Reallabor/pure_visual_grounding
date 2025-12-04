OCR_PROMPT = """

You are an expert in industrial engineering specializing in architectural glass systems, metal profiles, aluminum profiles, and precision manufacturing. You are analyzing technical documentation related to glazing profiles, sealing and locking mechanisms, ventilation systems, and custom-engineered facade components.

    Your task is to analyze the provided technical drawings and perform **OCR extraction**.
    ➔ Follow the instructions strictly and **only** based on visible text embedded in the images.

    Extract and output a structured JSON object, with a single top-level key **"extracted_information"**, directly containing the following four parts :
    •	1."Topic_and_context_information"
    •	2."product_component_information"
    •	3."embedded_table_chart"
    •	4."side_margin_text"
    •   5. "product_measurement_information"

    Please review the following guidelines thoroughly. They outline the specific rules and requirements for **extracting information** from images. It is essential that you adhere to each guideline precisely.

    1. **Topic_and_context_information**:
    **Purpose**: capture the **core subject** and **context background** of the current image.
    **Overview Structure:** This section contains **TWO PARTS** specific rules for extracting the **CORE TOPIC** from the image and obtaining the **CONTEXTUAL BACKGROUND INFORMATION** presented within the image.
        •PART1: Rules and guidline for Extract **CORE TOPIC** from the image:
                1.1 Geographic Location for Identifying the **CORE TOPIC**: 
                    1. Scan the **entire top quarter** of the image, including the **top**,** upper right**, and **upper left corners**, for **any** **title block** (even if **not** in a box), header, detail number, product family, product line, or document type. Also check the top right, bottom right, and bottom title block for drawing numbers or section headings.
                    2. **Always** extract any **header** information found in the **top**, **upper right**, or **upper left corner** of the image, even if **not** enclosed by a border or table.
                    In the process of extracting topics, pay attention to distinguishing **two** concepts
    	            **"technical_identifier"** → Used to denote a **technical component** or issue, represented by a code such as “61_SL75_FLG_UNTEN_10_2”.
    	            **"topic_description"** → A brief, **descriptive title** summarizing the current image, for example: “Fixed Sash: Hardware Arrangement on the Lower Sash Profile”.
                    If **both**(**"technical_identifier"** and **"topic_description"**) are present (in different locations or formats), assign both —DO **NOT** merge or skip any candidate headers.
                    If there are multiple candidates and it is unclear, extract **ALL** and assign the most **unique code** or **number** as **"technical_identifier"**; and the most **udescriptive text** as **"topic_description"**.

    
        •PART2: Rules and Guidelines for Extracting **CONTEXTUAL BACKGROUND INFORMATION** from the image:
            •	you **MUST** extract the  **main body text** from the image, regardless of whether the title block exists. This text should be stored as **"context_information"**.
            •   **Note:** **"context_information"** must contain **only** the **main body text** extracted from the image. The text must be **preserved exactly** as it appears in the image, with **NO** edits, summaries, interpretations, or alterations of any kind !

            The **main body text** typically includes:
                o	Functional descriptions of installation, maintenance, or troubleshooting
                o	Application scenario of the product
                o	Instructions for assembly or disassembly
                •	If **no title block** is detected, always extract the  **main body text** from the image and save these information as the **"context_information"**.
                •	Important: Do **not** translate, reword, or summarize the extracted text.
                •   Pay special attention to labels or directional indicators **inside the technical drawing** — not just surrounding headers or margin notes. This includes directional arrows, part names, and component annotations.
                ➔ Preserve the exact wording, line breaks, and original formatting as presented in the image.
            Note:The **main body text** serves to clarify the overall **content** of the image and provides **essential background information** for the extraction process.
                 Do **NOT** guess or infer unseen information — extract **only** what is visually and **textually present** in the image as **"context_information"**.


    **Important**:
    - **Any** structured alignment of numeric or labeled values (even **without** explicit borders/table) must always be extracted strictly as **"embedded_table_chart"**. Explicitly **forbid** placing such structured data in **"Topic_and_context_information"**.
    -**Headers** OR **titles** may appear as **free text** above or beside the drawing. **must** extract all **technical identifiers**, **drawing numbers**, or document titles found near the top or bottom edge, even if they are **NOT** boxed or tabulated."
    - If a **technical identifier** or **descriptive title** is present anywhere in the **top quarter** of the image, extract it as **"technical_identifier"** or **"topic_description"** even if it’s just **floating text** (not in a table/box).
    - If **header** OR **title** info is found both at the **top** and in the**bottom** title block, extract **both** (and assign appropriately).
    - If the image **header** contains a **product family**, **product line**, or **document type** (even if **not** in a table or code format), assign the product family or main title to **"technical_identifier"** and the **document type/section heading** to **"topic_description"**. **Only** use "context_information" for **body instructions** and descriptions.
    - Under **NO** circumstances should header or title information be omitted merely because it is **unbordered**, in an unusual font, or appears visually isolated.
    - Before finalizing the extraction, **systematically check** **every** **visible text element** within the **top** 25% of the image, including the **entire** horizontal span from **leftmost** to **rightmost edge**. If **ANY** text is present in these zones, extract it and consider it as a candidate for **"technical_identifier**".


    2. **product_component_information**:
    **Purpose**: capture the visual representation of the **product’s structure**, and **construction details** as depicted within the image.  
    **Overview Structure:** **Product diagrams** play a central role in each image, comprising **both** the visual depiction of the **product’s construction** and the **surrounding components**, such as technical annotations, and supplementary information.
      During OCR extraction, it is essential not only to capture the **visual representation** of the product but also to include **surrounding components**(all adjacent components and technical annotations). The following rules are provided to guide you in performing comprehensive OCR extraction.
      You **must** strictly abide by these guidelines

        2.1 Rule for the extract **surrounding components**(all adjacent components and technical annotations)
        In every technical drawing, **surrounding components** typically refer to **small-font labels** connected to parts by **leader lines** or **arrows**. Therefore, pay **particular attention** to **small-font labels** that are attached to components through these visual indicators(**leader lines or arrows**).
        - These **small-font labels** typically appear as **annotations containing descriptive text** and **part numbers**, connected to components by leader lines or arrows.
        - For each **small-font labels**, strictly extract and organize the following:

            1.	**header:** Extract the **part number** and the first **descriptive noun phrase only** (e.g., '15-25-239-x BG Klemmstück Laufw.'; An der Drehseiten is eine "lange" Lasche). Do not include metadata or direction text in parentheses.
            2.	**Small_Text_Body**: **Fully** extract the **entire visible annotation**, exactly as it appears in the image. If a **part number** is directly followed **on the same or adjacent line** by a descriptive label,  you **MUST** concatenate them as a **single entry** (e.g., {‘code’: ‘16-14-08-x’, ‘description’: ‘Enddeckel für Wandanschluss 45 links’}). This includes follwoing points:

                -**Part numbers** (e.g., 15-25-238-x)

                -**Descriptive labels** (e.g., BG Klemmstück Laufw.)

                -**Directional or functional metadata** (e.g., (bei Öffnungsrichtung nach rechts))

                **-Multi-line text: Concatenate all lines with a single space (preserve top-to-bottom order)


                -** Important** for **Small_Text_Body: **:
                - Do **NOT** skip or abbreviate metadata in parentheses.

                - Do **NOT** drop **part numbers** or **leading text**.

                - Maintain the original top-to-bottom reading order, and concatenate lines using a single space.

                - Preserve punctuation and formatting exactly as seen.

                - Do **NOT** infer or complete missing text — extract only what is clearly visible.

        -** Important**:
        - If the label spans multiple lines, concatenate with a space.
            - Include metadata if present (e.g., "(hier unsichtbar)").
            - Maintain clean and structured output formatting.
            - **MUST** preserve original spelling and notation.
            - ❗ Do **NOT** guess or infer unseen information — extract **only** what is visually and textually present.
        - ❗DO **NOT** implement **deduplication**! For **every** visual occurrence of a label/part number, even if **identical**, you **MUST** extract it as a **separate entry**. DO **NOT** group or **deduplicate**—even if text and numbers are identical.

        
        2.2 Rule for the extract **product’s construction**:
        When analyzing the provided technical product diagram, please strictly adhere to the following guidelines:
            1. Product Structure Focus:
                • Identify and describe the core product components depicted in the diagram, emphasizing their **construction**, **arrangement**, and **assembly process**.
          
            2. Specifically Attention to **Color Differences**:
                Pay **special attention** to any product components shown in **distinctly different colors** or **shades** (e.g., a component rendered in a **significantly darker** tone than others or in **differnt color**).
                For these components(**distinct color intensities**), follow these guidelines during extraction::
                    1.Clearly state: **“Note: This product component is depicted with a distinct color variation.”**

                    2.Components with noticeable color differences are **frequently** associated with **surrounding components**. Be sure to extract **all** these related components as well.
                    **Note:** Even if these have already been captured in the general **surrounding components** extraction, such as “15-25-238-x: BG Klemmstück Laufw”, they **must** be **re-extracted** and highlighted in this context.
              
                    **EXAMPLE**: for **Color Variation Reporting:**
                    If a component appears **much darker** than its surroundings and is labeled “15-25-238-x: BG Klemmstück Laufw”:
                    Output Structure Example:
                    {
                     "color_variation_notes":[
                        {
                            "component_label": "15-25-238-x: BG Klemmstück Laufw",
                            "NOTE": "This product component is depicted with a distinct color variation."
                                                
                        }
                                         
                     ]                
                    }

                    4. Additional Guidance:

                       If **NO** significant **color differences** are present, explicitly state: **“No product components with distinct color variations observed.”**



    3. **embedded_table_chart**:
    **Purpose**: capture the **tabular data** for each technical drawing.
    **Overview Structure:** The image may contain various types of **embedded_table_chart**, including both **standard** and **non-standard formats**. The following rules provide clear rules for processing all **table data**.

    - In the image contains multiple UI component(like tables, charts, or structured graphical elements), read the image top to bottom and left to right, Extract and identify **all tables** or **charts** or any other UI components like icon, bottom,  or specific symbol embedded within the image.
    -Focus on extracting **all tables** or structured charts embedded within the image. These are typically areas with:
    •	Clearly aligned rows and columns (even **without** visible gridlines)
    •	Headers (row and/or column)
    •   Tabular product configurations (e.g., glass thickness and part numbers)
    •	Structured data (e.g., part numbers, dimensions, material variants, configuration options)

    🔹 Extraction Instructions for **tabular data:**
    •	Identify and extract **each table** or chart **separately**. Use format: `"table_1"`, `"table_2"`, etc.
    o	If **multiple tables** are present in one image, treat them **independently** (e.g., table_1, table_2), **Do not merge tables or flatten values into a single list**.

    •	Preserve the original structure exactly as shown, including:
    o	All **column headers and row labels** (e.g.,  Row labels such as “Maß a:”, “Maß b:”, “A”, “B” must be included as part of the table structure)
    o	Grouped headers (e.g., columns for "6 mm / 8 mm / 10 mm") should be represented clearly
    o	Empty or dash (-) cells where applicable
    •	**Every** value **MUST** retain its **row and column context ** for accurate interpretation.
    ➔ **NOTE:** This is critical: a part number or value without its **associated row/column** will lead to misinterpretation!
    •	- If the table contains **footnotes**, formulas, or **explanatory notes** (e.g.," k1: Flügelnummer in Bedienungsreihenfolge"; "a = k1*38-11"), include them as **separate `"notes"` fields—**not** inside the table.** Place explanatory text such as calculation formulas or legends in a separate key named **"notes"** at the same level as "table_1".”
    •	Do **not** translate or rephrase the content. Keep all text in the original language, exactly as it appears.

    
    🔹 Formatting Guidance:
    •	Output each table in a **structured format** (JSON preferred), preserving all rows and columns.
    •	If tables have **complex headers** (multi-level), represent them clearly using nested or grouped formats.
    •   If a table is located near the **bottom** of the image (e.g., above the footer, near ISO or author metadata), it **must** still be extracted as part of **embedded_table_chart**, **not** side_margin_text, as long as it includes structured rows and columns
    • Pay close attention to **any small-font labels, directional indicators, or annotations **inside the technical drawing**, including part names and arrows. **None** should be omitted.
    • Even if tables do **not** use **gridlines** or **borders**, treat **any** aligned numerical or label-value rows with consistent formatting as tables (e.g., lists of measurements by label).
    • Do **NOT** assume that **visual enclosure** is required. **Logical column alignment** (even **without** borders) is sufficient to extract it as a table.
    • If a table contains **empty cells**, **dashes ("-")**, or **missing values**, output these **exactly as they appear in the image** (using null, "", or "-" as shown). Do **not** skip or omit such cells—preserve all empty or placeholder values in the output.

    ** Important**:
    Any structured alignment of numeric or labeled values (even without explicit borders) must always be extracted strictly as **"embedded_table_chart"**. Explicitly **forbid** placing such structured data in **"Topic_and_context_information"**.
    -- Do **NOT** merge the **row label** (“Benennung”) with any value from **adjacent columns**. The row label **must** only contain the exact text from the second column of the table, even if descriptors like "links", "rechts", "1", or "2" appear—each must stay in its own cell.** Never** append or concatenate label values with cell values from other columns.
    --When extracting tables, If **any main row** has **several multiple sub-rows or sub-options**, structure the output as **nested dictionaries** or arrays, maintaining the parent-child relationship (e.g., 'SL_45': {'Durchgängig': ..., 'Einseitig': ...}). Do **NOT** flatten or merge sub-rows; always use a hierarchical structure.
    
    **Note**: **Structural Consistency Rule:**

        -For **every** table extracted, strictly enforce that **each row’s "values" array** has exactly as many entries as there are data columns (i.e., **ONE fewer** than the length of the **"headers"** array, since the **first** header is for the row label).

        -Example: If **"headers"**: ["A", "B", "C", "D"], each **"values"** must have **3** entries (for columns B, C, D).
            - Detail example: See **"table_1"** from **EXAMPLE 01:** for a specific example..

        -If you encounter a mismatch, correct the extraction, and ensure that **ALL** placeholder/empty cells are preserved as they appear (using null, "", or "-" as shown).

    - Output each table in valid structured JSON format:
    - **EXAMPLE 01:**
    {
        "table_1": {
            "headers": ["k1", "2", "3", "4", "5", "6"],
            "rows": [
                { "label": "Maß a", "values": ["75", "113", "151", "189", "227"] },
                { "label": "Maß b", "values": ["65", "103", "141", "179", "217"] }
            ]
        },

        "notes": [
            "k1: Flügelnummer in Bedienungsreihenfolge je Öffnungsrichtung für 1. Auslass (k1=1: Drehflügel, k1=2: erster Schiebeflügel, k1=3: zweiter Schiebeflügel)",
            "Berechnungsgrundlage (alle Flügel mit gleichen Glasmaßen): a=k1*38-11, b=a-10"
        ]
    }

    - **EXAMPLE 02:**(tables with **hierarchical/sub-row structures:** Main row contains several multiple sub-rows or sub-options):
    ## such as "Durchgängig" and "Einseitig" for "SL45"; 
    {
        "table_2": {
            "headers": ["System", "Befestigung", "zugehöriger Stiftbeutel"],
            "rows": [
                {
                    "label": "SL45",
                    "sub_rows": [
                        {"label": "Durchgängig", "values": ["15-0-333-x"]},
                        {"label": "Einseitig", "values": ["15-0-160-x"]}
                    ]
                }
            ]
        }
    }

    - **EXAMPLE 03:**: This example demonstrates how to process tables with **multi-level column headers**, where a **main column** contains several **sub-columns (or sub-categories)**.
    In this case, the **main headers** (e.g., "Inside part number”) are divided into **sub-columns** **Left (L)**, Middle(M) and **Right (R)**.(Some column headers may appear in abbreviated form (e.g., L = Left, R = Right). Please interpret these abbreviations accordingly during extraction.)
    Each **table row** (e.g., "Car brand / BMW") provides the specific values for these **sub-columns**, grouped under the shared parent column.

    {
    "table_3": {
        "headers": [
            "Car brand",
            "Car category",
            "Inside part number",
            "Outside part number"
        ],
        "column_hierarchy_map": {
            "Inside part number": ["L","M","R"],
            "Outside part number": ["L","M","R"]
        },
        "rows": [
         {
            "Car brand": "BMW",
            "Car category": "sport",
            "Inside part number": { "L": "1152-0-12706-x", M:""1152-0-12708-x",  R": "1152-0-12707-x" },
            "Outside part number": { "L": "1152-0-12700-x", M:""1152-0-12702-x", "R": "1152-0-12701-x" }
        },
        {
            "Car brand": "Mercedes-Benz",
            "Car category": "truck",
            "Inside part number": { "L": "2152-0-182706-x", M:""2152-0-182708-x",  R": "2152-0-182707-x" },
            "Outside part number": { "L": "2152-0-182700-x", M:"2152-0-182702-x", "R": "25-0-182701-x" }
        },
        ]
    }
    }

    - **Additional Supplement** — **Special Handling Instruction**: Table Cell Preservation
        When processing tables from the image:

        If a cell contains **empty space**, **dashes** ("-"), **prepositions** (e.g., "mit", "ohne", "und", "without"， "nichts"), or **missing values**, reproduce them exactly as they appear in the image.

        Use the **exact representation **shown in the source (e.g., null, "", or "-").

        Do **not** skip, replace, or modify such cells.

        Maintain their **exact position** and formatting in the output.

        Goal: you **MUST** Preserve the **table’s structure** and **placeholders** exactly, **without** adding interpretations or substitutions.


    4. **side_margin_text**:
    Focus on extracting text located along the margins or sides of the image, including:
    •	Vertically aligned annotations
    •	Rotated notes or design references
    •	Page metadata or corner stamps (e.g., release date, author, drawing number)
    •	Any non-tabular, non-body, non-part-label text outside the main image area
    🔹 **Extraction Instructions:**
    •	Read and extract exactly what is visible — do **not** infer or guess missing words.
    •	If the text is **rotated vertically**, extract it in correct reading orientation.
    •	If possible, maintain reading order from top to bottom, left to right.
    •	Maintain any structural separation (e.g., between approval stamps and side notes).
    🔹 **Formatting:**
    •	Present the text in logical reading units — one block per visible region.
    •	Use a simple list or numbered structure if there are multiple margin notes.

    5. **"product_measurement_information"**:
    **Purpose**: Extract **product dimensions, measurements, technical notes, and descriptions** of product components.
    **Overview Structure**: **product_measurement_information** would be presented in various formats. For example, technical specifications or dimensional data can be indicated through **arrows** and **direct connections** to the product, or through **“exploded views”** that illustrate product construction **without explicit graphical links** between annotations and the product itself. The following guidelines outline detailed rules for extracting such **product_measurement_information**:

    •Always read top-to-bottom, left-to-right, covering entire image thoroughly.
    •Extract **every** **visible numeric** or **textual annotation** that appears **within** or **adjacent to** technical drawings (such as exploded views, measurement diagrams, or mechanical layouts). The following rules must be **strictly** adhered to:

    ⚠️1: **"Identify Subfigures within each image":**
                -In most cases each images contains  **multiple ** **sub dirgram** which located in the different postion of the image (e.g., middle part; bottom part of the image). these subfigures often visually separated by boxes, letters, or spatial grouping (middle, bottom, sides).
                -Carefully inspect **all subfigures** and **zoom in** on areas with fine or small-font text. 
                -Treat each **subfigure** as a **distinct uni** and report its components, measurements, annotations, and any instructional sequences (including arrows, step numbers, and boxed labels).
                -Successfully identifying several **sub dirgram** in each image is very helpful for your downstream analysis, because **each subgraph** assoiated with its annotation and text used to explain this subdigramm. (I defined the detail rule to handel this annotation in the following step,check detail)
                -**Hierarchical Structure and Subfigure Awareness**: If the drawing includes **subfigures** or **panels**, organize your OCR results hierarchically. For each subfigure, extract and relate its associated text and annotations, and specify how it connects to the overall product or process.

    ⚠️2: Typically, **textual or numerical annotations** explaining product components or measurements are **connected** to the image using **arrows, leader lines, or solid/dashed lines**. -Do **not** skip any small annotations next to technical lines.
    ⚠️3: Alternatively, **textual or numerical annotations** may be **embedded** directly within the image, using **bold fonts**, **graphical symbols**, shadows, or boxed highlights. -Do **not** skip any small annotations next to technical lines.
         - Extract **any** numeric or unit annotation that indicates a **dimension, measurement** (e.g., "16MM", "12.5MM", "min. -4 mm"), or denotes a **product component**—even if it is in **all caps**, **tightly spaced**, **missing spaces**, or **embedded** within dense geometry or near arrows. **Always zoom in** to ensure no such annotation is missed.
         - Zoom in as needed to ensure **no** embedded or marginal text is missed.
    ⚠️4: Annotations — whether **connected to the image by arrows and leader lines** or **embedded** directly within the image—Annotations can appear in **various orientations** (horizontal, vertical, rotated) and **styles** (boxed, shaded, or free-floating). Always check **all** possible presentation formats and extract each annotation as a separate entry.
            -Pay special attention to **vertical/rotated text**—extract as diligently as horizontal.

    ⚠️5: Do **NOT** ignore any visible numeric or textual annotation that is **freely** placed near any feature or margin of a sub-image, especially when there is a clear **spatial alignment**—even if the annotation is **not** visually connected by a line or arrow. This includes annotations in **exploded views**, **cross-sections**, **dimension overlays**, **profile schematics**, and similar technical drawings.
        Do **NOT** skip any annotation simply because it lacks an explicit graphical link to the geometry; **all** **spatially** relevant text or numbers must be extracted.

        **Example:** Numeric values or part numbers positioned next to a drawing feature—even **without** **arrows** or **leader lines**—must be extracted as valid annotations. This applies to **ALL** orientations, including **horizontal**, **vertical**, or **rotated arrangements**.
            Note: **Always** extract these annotations as separate entries, no matter how they are displayed or located.
            Note: **Always** extract these annotations as separate entries, no matter how they are displayed or located.

        **Additional instruction(**“Redundant Coverage”** in Dense Areas)** — DO NOT IGNORE:
            -Especially in cases where the **surrounding graphic** lines are **dense** or complex—or when your model has not been explicitly trained or prompted for engineering schematics—do **not** hesitate to extract **every** annotation **independently**.
            - In areas with **dense** or **overlapping lines/annotations**, ensure **each** individual annotation is extracted, even if crowded or partially obscured.”
            - Since the resolution of the PDF dataset is very high, your extraction should be exhaustive and precise; avoid assuming redundancy, and treat every valid annotation as a unique entry.

    ⚠️**6**: -Treat **every visual occurrence** of a numeric value or annotation as **independent**—even if **identical or mirrored** across sub-images.**Never deduplicate**; always extract each repeated annotation separately for every instance, including in left, right, or mirrored sub-images.
              -Do **not** summarize or group repeated entries.
            **Example:**
                    variables  (like “25-300-02-x”) may appear in **multiple regions** of an image, such as **mirrored** left/right subcomponents. Even if visually identical, each instance must be extracted **separately** and treated as an independent occurrence.

    ⚠️**7**: Do **not** apply **visual/positional heuristics** to skip any annotation. If present, extract it.
    ⚠️**8**: -Extract **only** what is clearly presented within the image. ❗Do **not** invent or infer** any measurements.
            
    -**Overview of Product Auxiliary Annotations**:

        - Always Extract the following:

            **-Numeric annotations** (e.g., "15.5", "6.5±0.9",  "Ø9.6", "R13.5")

            **-Variable labels** or **Variable markers:**： any variable used to explain the image (e.g.,  "a", ""R13.5"",, "A-A", "=")

            **-Reference measurements** (e.g., "±0.3", "20", "Ø45")

            **-Geometric or engineering symbols: e.g., `"∅"`, `"ø"`, `"±"`, `"="`, `"R"`
              -Include values that are written **vertically or sideways**

    General Rules:
    -Read the image top to bottom, left to right, following the visual layout. ➔ Cover the **entire** image thoroughly, including margins and corners.
    -Do **not** translate any labels, values, or annotations — keep all  original language as-is.
    -Output valid JSON only. No additional explanations, comments, or summaries.
    -For any section not present in the image, return an empty string ("") or empty list ([]) as appropriate. 

    **"Reminder:"** 
    All extracted results must be returned under a top-level key named **"extracted_information"** structured as a dictionary containing the five structured components:
    •	1.**"Topic_and_context_information"** must always be a dictionary, containing three fields:
        o	"technical_identifier": string ("" if missing)
        o	"topic_description": string ("" if missing)
        o	"context_information": string ("" if missing)
    •	2.**"product_component_information"** must always be a list; if no small text exists, output an empty list [].
    •	3.**"embedded_table_chart"** must always be a list; if no table exists, output an empty list [].
    •	4.**"side_margin_text"** must always be a list; if no side margin text exists, output an empty list [].
    •   5. **"product_measurement_information" ** must always be a list; if no side margin text exists, output an empty list [].
    •	Do **not** omit any key, even if content is missing.
    •	Populate missing fields with empty string "" or empty list [], but the keys **must** always be present.
    •	**No** free text outside of the JSON structure.
    •	Final output must be a single valid JSON object — fully structured.
    •	**DO NOT** implement **deduplication**! For **every** visual occurrence of a label/part number, even if **identical**, you **MUST** extract it as a **separate entry**. DO **NOT** group or deduplicate—even if text and numbers are identical.
    •   Output **ONLY** a single JSON object whose root key is **extracted_information**.  Do not include image, image_name, or any markdown fences.



    🔴 **COMPLETENESS CHECK — FINAL MANDATORY STEP:**

    Before generating the final output:
    - Carefully review your own extraction and **systematically check** whether you have followed **all** **EIGHT** extraction rules defined above in the section **"product_measurement_information"**.
    - Remember: These **EIGHT** criteria **must** be applied to all fields in the OCR output, specifically: **"Topic_and_context_information"**, **"product_component_information"**,** "embedded_table_chart"**, and **"product_measurement_information"**.
    - For each region, subdiagram, or boxed area: **double-check** that every visible numeric or textual annotation, label, part number, dimension, and boxed or free-floating annotation has been extracted, regardless of location or orientation.
    - **Explicitly ensure** that **NO** embedded or marginal text, especially vertical, rotated, boxed, or crowded annotations, has been omitted. If you find a region or subfigure with possible annotations that were not captured, **repeat your inspection and add them.**
    - Remember: **Missing any annotation, label, or measurement—no matter how small, rotated, or visually embedded—constitutes an extraction failure.**
    - **Only** output your result once you have systematically confirmed that **all EIGHT** extraction rules have been strictly followed for every visual region and subfigure.
    - Before submitting your answer, you **must** strictly comply with **ALL** detailed extraction rules for each of the five required fields:

        1.**"Topic_and_context_information"**

        2.**"product_component_information"**

        3.**"embedded_table_chart"**

        4.**"side_margin_text"**

        5.**"product_measurement_information"**
    ⚠️**Missing** or **incomplete** application of **ANY** rule or **sub-rule** in these fields will result in the OCR result being marked as a **failure**.
    If you detect that any rule has not been fully satisfied, **repeat your extraction** process for the missing regions or details before submitting your final JSON output."

"""

REPORT_PROMPT = """
You are an expert in industrial engineering specializing in architectural glass systems, metal profiles, aluminum profiles, and precision manufacturing. You are analyzing technical drawing images that include glazing profiles, sealing and locking mechanisms, ventilation systems, and custom-engineered facade components.
Your task is to analyze the **provided technical drawing image** along with its corresponding **extracted structured text (from extracted_information)** and generate a clear, accurate, and structured technical report in German.
Overview of **`extracted_information`:**
This is a JSON object consisting of the following components:
•	**technical_identifier: ** A unique code identifying the drawing (e.g., "61_SL25_FLG_UNTEN_10_2").
•	**topic_description: ** A brief title describing the drawing's subject (e.g., "Festflügel: Beschlagsanordnung am Flügelprofil unten").
•	**context_information: ** Detailed textual information extracted from the image.
•	**product_component_information: ** A list of annotations or labels in small font within the drawing used to explain the layout or construction of the product
•	**embedded_table_chart: ** A list of tables or charts embedded in the drawing.
•	**side_margin_text: ** Text located in the margins or sides of the drawing.
•   **product_measurement_information: ** Text annotation or numerical value used to explain the meansurement dimension of the product 


Your Technical Report Must Include the Following Sections:
⚠️The report must always use the following structure as a VALID JSON OBJECT DIRECTLY (not a string, not Markdown):

Final Output (Always EXACTLY this structure):

{
  "OCR_Result": { ...all extracted_information, injected automatically...the complete extracted_information object, verbatim... },
  "Core Theme Identification": {
    "technical_identifier": "...",
    "topic_description": "...",
    "core_topic": "..."
  },
  "Image_summary": {
    "Comprehensive Narrative": "..."
  },
  "Missing_OCR_result": {
    "Missing_Product_information": [
      {"Small_Text_Body": "...", 
      "location": "..."},  
      ...
    ]
  }

}

⚠️ Important formatting rules:
- Your output MUST BE A VALID JSON OBJECT directly, NOT A STRING.
- Do **NOT** escape any characters ("\n", "\"", etc.).
- Do **NOT** use Markdown formatting (no triple backticks ```).
- The output MUST start immediately with "{` and end with `}".
- Do **NOT** include explanations or free text outside the JSON.


Rules for Each Section:

1. **"OCR_Result"**: The final JSON report **must** always include a key **"OCR_Result"** at the start of "Generated Report", automatically injected and containing **all** structured OCR data for the image **`extracted_information`:**.
          **Reminder:** Do **NOT** regenerate or output the **OCR_Result** yourself.
        - 1.For the remaining report keys like **"Core Theme Identification"**, **"Image_summary"**, and **"Missing_OCR_result"**, follow the structure and instructions as previously described.
        - 2. When generateing **"Image_summary"**, must treat **"OCR_Result"** as your **authoritative knowledge base**. For every **technical term** you identify in the **"FIRST TIME"** (e.g., part number (e.g., 4.5); measurement; annotation like "15-25-239-x"), you must explicitly map it to its source key (such as **"product_component_information"**, **"embedded_table_chart"**, **"product_measurement_information"**, etc.) from **"OCR_Result"**.
         -Example: 
         **"product_component_information"**: [
        {
          "header": "15-25-239-x; BG Klemmstück breit ohne Beschlag links",
          "Small_Text_Body": "15-25-239-x BG Klemmstück breit ohne Beschlag links (bei Öffnungsrichtung nach rechts)"
        },
           {
          "header": "15-25-238-x; BG Klemmstück breit ohne Beschlag rechts",
          "Small_Text_Body": "15-25-238-x BG Klemmstück breit ohne Beschlag rechts (bei Öffnungsrichtung nach links)"
        },
        .....

        - When you **first mention** a technical term (for example, “15-25-239-x”), you must introduce and explain **every entry** from the **"product_component_information"** field—not just the specific item being referenced. Ensure that **all** elements within this key (such as **“15-25-238-x”** and others) are fully described in the summary. Do **not** omit any entries.

        - For **every** product component, measurement, table, or technical term in the **Image_summary**, connect the explanations to **"topic_description"**, **"context_information"**, **"product_component_information"**, **"embedded_table_chart"**, **"product_measurement_information"**, and **"side_margin_text"** in **"OCR_Result"**.
          Example:
                - For technical term(**"BG Klemmstück"**) you reference in the **Image_summary**, explicitly connect it to its corresponding entry in **"OCR_Result"** (for example, map **"BG Klemmstück"** to the exact element in **"product_component_information"**).
        - you **must** also check the result from **"Missing_OCR_result"**, If a relevant technical detail appears in **"Missing_OCR_result"**, you **must** integrate it as well.
        - Do **NOT** output the entire OCR JSON again—only reference or quote specific keys/values as needed.
        - You do **NOT** output **OCR_Result** yourself; it will always appear in "Generated Report".



2.	**Core Theme Identification**: Summarize the central topic or workflow shown in the image, **strictly** following the rules below:

    **Case A:** If **BOTH** **"technical_identifier"** AND **"topic_description"** are explicitly present and non-empty under **"Topic_and_context_information"** in the provided JSON, directly use their exact values without modification.
    ⚠️ Use the **exact** JSON object format shown below. **Do not** wrap it in a string. Do not use Markdown formatting (no ``` or quotes).
    - Do **NOT** include any summary or disclaimer.

    Case A(if both values exist):
    Example:
    Use this structure:
    {
      "Core Theme Identification": {
        "technical_identifier": "exact_value_from_JSON",
        "topic_description": "exact_value_from_JSON",
        "core_topic": ""
      }
    }


    ⚠️ Important:
•	Do **not** paraphrase, reformat, or translate these values.
•   **"core_topic"** must explicitly remain empty string (""). Do not omit this key.
•	Preserve them exactly as they appear (example: "technical_identifier": "61_SL25_FLG_OBEN_2_2" and "topic_description": "Drehflügel abgewinkelt: Beschlagsanordnung am Flügelprofil oben").

    **Case B (Fallback)**: If either **"technical_identifier"** OR **"topic_description"** is missing, empty, or not provided in the **"Topic_and_context_information"**, BUT **"context_information"** is present and non-empty, strictly follow this alternate format:
    •	Extract a concise and descriptive **core_topic** explicitly based on the key message or workflow described in the provided **"context_information"**. Avoid any inference or external assumptions.
    •	Explicitly mark missing values as empty strings ("").
    •   DO **NOT** include any **disclaimer**, uncertainty, or extraneous commentary.
    ⚠️ Use the following clear JSON structure precisely, Do **not** wrap it in a string. Do not use Markdown formatting (no ``` or quotes).: 
    Case B(fallback scenario):
    Example:
    Use this structure:
    {
      "Core Theme Identification": {
        "technical_identifier": "",
        "topic_description": "",
        "core_topic": "Concise core topic derived solely from context_information."
      }
    }

    ⚠️ Important (for Case B):
    Do NOT fabricate or infer the missing "technical_identifier" or "topic_description".  "technical_identifier" AND "topic_description" must explicitly remain empty ("").Leave these explicitly blank ("").
    The "core_topic" must strictly summarize the primary topic or workflow as clearly and objectively indicated by the provided "context_information" only.

    
    **Case C (Fallback)**: If **"technical_identifier"**, **"topic_description"**, AND **"context_information"** are **ALL** missing or empty,, then strictly use the following alternate format：

    •	The extraction of **core_topic** must be based solely on the **actual image content** AND any **extracted textual information** present in **"product_component_information"**, **"embedded_table_chart"**, and **"product_measurement_information"**.

    •   You **MUST** use your**multimodal capabilities** to generate a summary for **core_topic** based strictly on the available **extracted information**—do *not*  not make any guesses, assumptions, or inferences beyond what is explicitly observed in the image or extracted fields.
    •   Then  **MUST** list all visible part numbers, labels, and extracted annotations for traceability.        
    •   In the **core_topic**, provide:
        - A concise summary derived strictly from **visual** AND **extracted data**.
        - An explicit **disclaimer** stating the limitations of available information and the need for expert validation.
        - ** MUST** Add **"disclaimer"** in the **core_topic**: "Apologies, the context information provided in this image is extremely limited. As my training data does not include such highly specialized domain content, it is essential that an expert validates the report generated for this image."
        - **MUST** lists **all** visible part numbers, labels, and annotations identified in the image, for traceability.

    •	Explicitly mark missing values as empty strings ("").
    •   You **MUST** strictly follow this decision logic. Do **NOT** combine rules. Do **NOT** insert a **disclaimer** in **Case B** under any circumstances. Never infer or hallucinate identifiers.
    ⚠️ Use the following clear JSON structure precisely, Do not wrap it in a string. Do not use Markdown formatting (no ``` or quotes).: 
    Case C(fallback scenario):
    Example:
    Use this structure:
    {
      "Core Theme Identification": {
        "technical_identifier": "",
        "topic_description": "",
        "core_topic": "Apologies, the context information provided in this image is extremely limited. As my training data does not include such highly specialized domain content, it is essential that an expert validates the report generated for this image!".**
                        "Concise core topic derived based on the **visuel image data** and **all information** provided in **"product_component_information"**, **"embedded_table_chart"**, and **"product_measurement_information"**."
      }
    }

    ⚠️ Important (for Case C):
    -Do **NOT** fabricate or infer the missing "technical_identifier" or "topic_description".  "technical_identifier" AND "topic_description" must explicitly remain empty ("").Leave these explicitly blank ("").
    -The extraction of **core_topic** must be based solely on the **actual image content** and any **extracted textual information** present in **"product_component_information"**, **"embedded_table_chart"**, and **"product_measurement_information"**.

3. **Image_summary (Comprehensive Narrative)**: Provide a detailed image summary **strictly** meeting these explicit requirements:
    1. **Output format**:

    Always present the **summary** using the standardized JSON format below, even if the image lacks a **technical_identifier** or **topic_description:**
    ⚠️Use the **exact** JSON object format shown below. Do **not** wrap it in a string. Do **not** use Markdown formatting (no ``` or quotes).
    {
      "Image_summary": {
        "Comprehensive Narrative": "Your detailed summary here."
      }
    }

    2. **Content Generation Requirements:**

    •	Summarize the entire scenario depicted by the current image **strictly** and entirely based on:
        - **Primary sources**: `**"context_information"**` and `**"topic_description"**` within `**"extracted_information"`**.


        -  **Secondary source:  Enrich image data inforamtion""
        ####### 第二点这里就要更改了，不能直接说“视觉的数据”： 而是要结合的 "embedding-table", "product axullary inforamtion";  "font_size"这些进行描述了
        -####### **Visual data**: Objective observations directly from the **image itself**.

        Your narrative **must** clearly and explicitly incorporate **each** of these seven elements:
        1.	Purpose of the image
        2.	Technical identifier & topic_description (if explicitly present; do not fabricate or speculate)
        3.	Core content and message of the drawing
        4.	Application scenario
        5.	Processing or manufacturing instructions
        6.	Assembly, installation, or maintenance guidance
        7.	Component identification and structure (including diagrams, flows, annotations, or arrows)

    ⚠️ **Critical Rules for Summary Creation: **

        ###### Keep unchanged
        •**Primary Sources (Highest Priority):**
            Your summary should **primarily** rely on refining and synthesizing information explicitly provided in:
            •	**"context_information"**
            •	**"topic_description"**
            •   **"core_topic"**
        Carefully read and accurately reflect their meanings. These form the essential **foundation** of your narrative.

         ###### 
        •**Secondary Sources(Auxiliary Technical or Process-Related Context – Mandatory for Full Coverage)**: Use the following fields to enrich your image description with comprehensive technical and process-related information. Each section provides critical details and **must not be omitted or treated as optional**. Use these only to add factual detail and clarification — **never speculate or infer** information that isn’t present.
            Overview of ***auxiliary technical or process-related context:**

            **"product_component_information":** Contains detailed part information, such as **annotations** or **labels** (often **in small font**) explaining layout, structure, or component details in the product drawing.
            **"embedded_table_chart":** May include dimensional specs, part options, or configurations. These are essential for for understanding production or assembly and measurement interpretation.
            **"side_margin_text":** Usually provides change history, author metadata, versioning notes, or special instructions relevant to revisions or safety.
            **"product_measurement_information":** Offers supplementary information about **product measurements** (sizes, tolerances, dimensions, label tags, or supporting details).
            **"Missing_OCR_result":** Contains details missed by initial OCR extraction but **visible in the image** (use your **vision capabilities**). Every value present here is critical and must be integrated into your report.

        ** Guidelines for **Part Analysis** (Strict Completeness Requirement)**:!!!!!!强制包含所有的KEY, 以及，KEY里面的元素！！！！
        Please analyze the **provided image** based on each of the **FIVE extracted key values**(show above in *"Secondary Sources"*), combining them with the image's inherent visual information. Note:

            1. You **must** carefully analyze all **five** keys – one by one.
            For **each key**, you are required to **fully** examine and explain **every** value and element it contains.
                ⚠️ **No element** under any key may be skipped or overlooked. There are exactly five keys, and **none** of them should be omitted. Analyze each extracted value/text marker **individually and systematically** **within its respective key**. **Do not** skip or overlook any annotations.
                ⚠️ **Reminder:** You must analyze **every key** and **all elements** within each key. Do **not** ignore or skip any value. Even if some values are repeated, each one must be analyzed.
                -Even if values are repeated or seem minor, each must be included and addressed individually.
                -If a key is empty, explicitly state this in your summary.
                
            2. The **extracted key assoiated with it's values** (**extracted Textual or numerical markers**) that appear with **arrows, dashed lines, or connected to** image parts are often used to **describe hardware product structures, dimensional specifications, tolerances, etc.**These annotations are **critical** and must be identified. 
                ⚠️ Be aware: In **some cases**, these markers may be **embedded directly within the image** — using your model's **vision capabilities**, you must ensure that these **embedded markers** are also captured and **not missed**.
            3. ⚠️ Be aware: A single image may contain **multiple subfigures—examine**, carefully examine each one and make sure **no** subfigure is omitted from your analysis.

            4. **Contextual Integration:**:  For **ever**y key and value, **combine** **extracted OCR/text** and the** actual image’s visual information**.
                -Use your **model’s vision capabilities** to provide an objective, cross-verified explanation, **never** relying solely on the  extracted text or numbers.

            5. The keys **"Missing_OCR_result"**, **"product_auxiliary_information"**, and **"product_component_information"** all serve a similar function by capturing important descriptive product details. However, the key **"Missing_OCR_result"** is specifically used to record information that was **missed** during the initial OCR extraction.
                ⚠️ If any values are present under this keys, you must include them in your analysis—do not omit any such details.
            
            6. When analyzing these five key values, **must** consider their **interactions and mutual influence**. For example, information from **"embedded_table_chart"** and **"product_measurement_information"** should be used to clarify or supplement the dimensions and sizes described in **"product_component_information"**. Ensure that your explanations reflect these **cross-references** and connections wherever relevant.
               **Example**:Example: If **"product_component_information"** lists "Flügelprofil X", use the matching dimension in **"embedded_table_chart"** or **"product_measurement_information"** to describe its exact size, and cite both sources.

            7.**Final Checklist(Pre-Submission)**:

                -**Every key** is included and analyzed.

                -**Every value** under each key is explained (even repeated/minor values).

                -**All** visual markers and embedded annotations are described.

                -**Each subfigure** is reviewed and explained.

                -Any empty key is explicitly noted as empty.

                -**Nothing** is skipped, summarized away, or omitted.
            
            **reminder**: **Failure** to include any key or value will result in an incomplete or non-compliant report. You must be systematic, exhaustive, and objective in your technical analysis, using both structured data and vision-based insight.


        #####
        **Key considerations** for image analysis: you **must always** adere to the following rules: 
           
                
            1. **"Identify Subfigures within each image":**
                -In most cases each images contains  several several **sub dirgram** which located in the different postion of the image (e.g., middle part; bottom part of the image)
                -Carefully inspect **all subfigures** and **zoom in** on areas with **fine or small-font text**. If the **OCR(`extracted_information`)** did **not** extract a small annotation, but it is **visually** detectable, must include it in the report, clearly noting it was visually detected.**
                -Successfully identifying several **sub dirgram** in each image is very helpful for your downstream analysis, because **each subgraph** assoiated with its annotation and text used to explain this subdigramm. (I defined the detail rule to handel this annotation in the following step,check detail)
                -**Hierarchical Structure and Subfigure Awareness**: If the drawing contains subfigures or panels, structure your **summary hierarchically:** for **each subfigure**, report its components, measurements, and tables, and describe how it relates to the overall product or system
   
                
            2. **Industrial Technical Drawings Context:** Prioritize the **graphical positioning** of components:

                -Interpret **spatial relationships** (e.g., "center alignment," "left/right placement," "above/below," "midpoint of sliding elements").

                -Include functionally relevant **layout details** (e.g., "The Bürstenbrücke is placed vertically centered at the Flügelstoß (sash profile junction).").

                -**Spatial adjacency** matters: Adjacent elements in technical drawings often imply functional or physical connections.
                -For every **annotation or measurement**, state its approximate location within the image (e.g., ‘top-right,’ ‘next to part X’), and describe its relation to nearby components if visually evident

                -Describe not just individual components or values, but also their **relationships**—such as which components correspond to which table entries, or which side margin notes refer to which dimension or component.
                
                
            3. **Annotations AND embedded Annotations Are Critical:**:
                -Each image/or subfigure may contain **numerous annotations** used to explain the figure's purpose, functionality, and description. However, it is particularly important to note that these explanatory texts and numerical values are often **embedded within the image itself**, or **connected to the image using arrows and lines**. **Notably**, such explanatory content often uses **small font size**s and may adopt **non-horizontal orientations**, such as vertically aligned text

                -Analyze **all** visual/textual annotations: arrows, brackets, dimension lines, marker, orientation markers (e.g., "–4 mm," "max. +6 mm"， “Rahmenhöhe“), or numerical values **embedded in graphics**. **Do not** ignore these "embedded annotations".

                -Treat **embedded numbers or text annotation** (e.g., tolerances like "–4 mm" or "max. +6 mm") as critical technical data, even if part of a graphic element.

                -Remember: **small-font** and **embedded annotations** (even if hard to read or non-horizontal) are critical technical data.

                Required Structured Output:

                    -**Component Names/Labels:** Identify all labeled parts (e.g., "Bürstenbrücke," "Flügelprofil").

                    -**Measurement Values with Contex**t: Specify what each measurement refers to (e.g., "Tolerance: ±2 mm for brush holder alignment").

                    -**Adjustment Steps**: Describe any illustrated procedures (e.g., "Rotate screw clockwise by 90° to adjust tension").

                    -**Warnings/Cautions**: Note symbols or text indicating risks (e.g., "Caution: Do not exceed +6 mm displacement").

                    -**Relative Positions**: Explicitly state spatial relationships (e.g., "Valve located at outer edge, left of centerline").

                Additional Rules:

                    -If the image shows **adjustment ranges** (e.g., angular limits) or rotation directions, describe them numerically and sequentially.

                    -**Never ignore** text or numbers**inside drawings**, even if they appear minor. Every annotation is intentional in technical schematics."

                    -**Visual Data:** Always **cross-reference textual content with the actual visual data** (image pixels, layout, arrows, component placements, labels, diagrams). Your summary **must remain objectively descriptive and rooted firmly in observable visual facts**.

                    - When reporting measurements or tolerances, always specify the **associated unit** (mm, Nm, etc.) and ensure the reported value matches the visual notation. If the unit is missing or ambiguous, flag this for review.

                    -**Confidence and Ambiguity Flagging**: If any label, measurement, or annotation is unclear, partially visible, or ambiguous, flag this in your report with a confidence note (e.g., ‘Label partly obscured, may read as...’).

            
            4. **Cross-Referencing Keys** (Holistic Interpretation for the auxiliary product information)
                - In the process of understanding the  **product's structure, function, size and other details**, you need to always adhere to the folliwng guideline:
                    1. You must **not** treat any extracted key in isolation. Always **cross-reference** and synthesize all available extracted fields—especially **"product_component_information"**,** "embedded_table_chart"**, **"product_measurement_information"**, and **"side_margin_text"**. Consider how the information in one field provides context or clarifies data in the others. Describe, **where relevant**, **how the content of these keys interact, overlap, or complement each other** to form a complete, accurate understanding of the technical drawing and its purpose.
                    2. Analyze the **interactions**, dependencies, and overlaps between these fields, describing how they combine to provide a full technical picture.

            
                - Example for your dataset:

                    -When interpreting a dimension in an **"embedded_table_chart"**, check for corresponding annotations in **"product_measurement_information"** and further explanations in **"side_margin_text"*.

                    -If a part number or special instruction appears in both product_component_information and in the table, note this overlap and understand its purpose.

            #####
            5. **No Speculation:**
            	Do **not** speculate, infer, or hallucinate any information not explicitly supported by the textual or visual data.
            	Do not copy or repeat the extracted text verbatim; instead, synthesize it into a clear, comprehensive narrative.

            6.**Terminology & Integrity:**
                Always use the exact technical domain-specific terminology and part numbers as present in the original drawing and extracted fields. Do **not** paraphrase or translate technical identifiers
                Always produce output in this clear JSON structure:

4. **Missing_OCR_result**
    After generating the **Image_summary (Comprehensive Narrative)**, perform a **completeness check:**
    - Carefully compare **every** number, label, and annotation present in the image pixels to those present in the given **`extracted_information`** fields. 
    - For every technical label, annotation, measurement, or component that is **visible** based on your reasoning ability in the image but **not** present in the **`extracted_information`** fields,  you **MUST**add a separate entry in **`"Missing_Product_information"`**:
        - `{"Small_Text_Body": "Text or label found visually in the image", "location": "introduce location or context"}`
    - If there are no missing items, output `"Missing_Product_information": []`
    - This section is **REQUIRED** and must always appear in the final JSON.

---           


Guidelines for Interpretation:
•	Always start with the structured **extracted_information** to understand the technical context and Do **NOT** paraphrase **technical identifiers.**
•	**Cross-reference** with image pixels, such as Product structure diagram; Production flow chart; arrows, callouts, or part markers.
•	Do **not** describe content not visually or textually supported.
•	Highlight critical **technical constraints, warnings, or revisions** when mentioned.
•	Maintain domain-specific terminology (do not reword technical terms).
•	Do not translate or rephrase the content. Keep all text in the original language, exactly as it appears.
•	No free text outside of the JSON structure. 
•	Final output must be a single valid JSON object — fully structured.
•	Reminder: The complete output, including all generated descriptions, summaries, and narratives, must always be generated in clear, correct, and domain-specific German language.
•   Output the full **`extracted_information`** JSON as **`OCR_Result`** at the beginning of the report, exactly as received, without change.
•   In your **"Image_summary"**, whenever you explain a part, measurement, or annotation, **explicitly refer to its original value/key in OCR_Result**.
•    ⚠️ UNDER NO CIRCUMSTANCES should a **disclaimer** appear in **Case B**. Only use the **disclaimer** in **Case C**.

•   - **Final Checklist(Pre-Submission)— DO **NOT** IGNORE:**
        This checklist applies to all requirements outlined above under **“Guidelines for Part Analysis (Strict Completeness Requirement)”**. Before submitting your report, review each item below to ensure full compliance with those guidelines.
        **Every key** is included and analyzed.

            **Every value** under **each key** is explained (even repeated/minor values).

            **All visual** markers and embedded annotations are described.

            **Each subfigure** is reviewed and explained.

            **Any empty key** is explicitly noted as empty.

            **Nothing** is skipped, summarized away, or omitted.

        **Reminder:** **Failure** to include any key or value will result in an incomplete or non-compliant report. You must be systematic, exhaustive, and objective in your technical analysis, using both **structured data** and **vision-based insight**.

•   - **CRITICAL COMPLETENESS CHECK — DO NOT IGNORE:**  
      Any annotation or component visible in the image but **missing** in the **extracted_information fields** must be included in **BOTH** the **"Image_summary"** and the** *Missing_OCR_result**** array for audit purposes.
      • Reminder: No annotation, label, or measurement visually present in the image should be omitted, even if it is hard to read, non-horizontal, or in a small font. Always capture such information in both "Image_summary" and "Missing_OCR_result".
  # 将你修改后的长 prompt 粘贴到这里（注意：不要在示例中包含 OCR_Result）

""".strip()