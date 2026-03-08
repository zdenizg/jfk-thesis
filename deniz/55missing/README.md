# 55 Missing Documents — Analysis

This folder contains scripts and outputs for processing the 55 JFK documents that were missing from the main dataset.

## Contents

- `missing_file_ids.txt` – The 55 document IDs
- `check_missing_ids.py` – Verifies which of the 55 PDFs are present in the local `jfk_pdfs/` directory
- `ocr_missing_google.py` – Runs Google Cloud Vision OCR on the 55 PDFs with retry/backoff and checkpointing
- `categorize_55_missing.py` – Classifies each page using Groq (document type, OCR difficulty, handwriting, etc.)
- `ocr_missing_output_google/` – OCR text outputs (one `.txt` per document) and `summary.csv`
- `results/` – Categorization output (`jfk_categorization_55missing.csv`) and checkpoint file
- `jfk_pdfs/` – The source PDFs (not tracked in git; too large)

## Running

```bash
# Check which PDFs are available locally
python check_missing_ids.py

# Run OCR (requires Google Cloud credentials)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
python ocr_missing_google.py

# Run categorization (requires GROQ_API_KEY in .env)
python categorize_55_missing.py
```
