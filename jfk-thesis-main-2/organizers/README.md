# Document Categorizer

Categorizes JFK documents using **Groq Llama models** and **Leo Prompt Optimizer**.

## Features Extracted

| Feature                   | Description                                       |
| ------------------------- | ------------------------------------------------- |
| `includes_handwriting`    | Boolean – handwritten notes or signatures present |
| `has_shadowy_background`  | Boolean – dark or unclear background              |
| `document_quality`        | Quality score based on readability                |
| `text_density`            | Estimated word density per page                   |
| `has_stamps`              | Presence of stamps or marks                       |
| `has_redactions`          | Whether redacted text exists                      |
| `has_forms`               | Contains structured form elements                 |
| `has_tables`              | Contains tabular data                             |
| `is_typewritten`          | Indicates typewritten content                     |
| `paper_condition`         | Physical condition assessment                     |
| `primary_characteristics` | Combined summary of key attributes                |
| `document_type`           | Category classification (see below)               |
| `ocr_difficulty`          | Rated as "simple", "average", or "complex"        |

## Document Categories

- `classified_memo`
- `security_form`
- `personnel_record`
- `operations_roster`
- `cover_notification`
- `clearance_request`
- `administrative_memo`
- `historical_record`
- `field_report`

## Prompt
```python
combined_prompt = """
<role>
You are a document analysis expert trained to extract structural, visual, and categorical insights from scanned or photographed documents.
</role>

<task>
Your task is to analyze a provided document image and return a structured JSON object containing detailed visual characteristics, category classification (from a fixed list), and OCR difficulty rating.
</task>

<instructions>
* Step 1: Visually inspect the document image for handwriting, shadows, stamps, redactions, forms, tables, and paper quality.
* Step 2: Assess overall document quality and text density.
* Step 3: Determine if the document is typewritten.
* Step 4: Assign descriptive tags summarizing key characteristics (max 5).
* Step 5: Classify the document using ONE of the following categories only:
  - "classified_memo"
  - "security_form"
  - "personnel_record"
  - "operations_roster"
  - "cover_notification"
  - "clearance_request"
  - "administrative_memo"
  - "historical_record"
  - "field_report"
  Do not generate any new or custom categories.
* Step 6: Rate the OCR difficulty as "simple", "average", or "complex".
* Step 7: Return ONLY the structured JSON output as specified—no markdown, no explanations, no extra text.
</instructions>

<output-format>
Return a single JSON object using this exact structure:
{
  "includes_handwriting": boolean,
  "has_shadowy_background": boolean,
  "document_quality": string,
  "text_density": string,
  "has_stamps": boolean,
  "has_redactions": boolean,
  "has_forms": boolean,
  "has_tables": boolean,
  "is_typewritten": boolean,
  "paper_condition": string,
  "primary_characteristics": array,
  "document_type": string,
  "ocr_difficulty": string
}
</output-format>

<user-input>
[Document image provided by the user]
</user-input>
"""
```

## Output

`document_categorization.csv` – Document features for all PDFs