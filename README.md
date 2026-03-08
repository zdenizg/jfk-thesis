# JFK OCR Thesis Materials

This repository contains tools and results for extracting and analyzing text from JFK assassination documents released as part of the 2025 records release.

## Repository Structure

```
.
├── ocr_missing_google.py           # Google Cloud Vision OCR script for the 55 missing PDFs
├── categorize_55_missing.py        # Classifies each page using Groq (type, difficulty, handwriting, etc.)
├── check_missing_ids.py            # Verifies which of the 55 PDFs are present locally
├── missing_file_ids.txt            # List of 55 document IDs missing from the main dataset
├── requirements.txt                # Python dependencies for the OCR script
├── Thesis_Proposal.pdf
│
├── ocr_missing_output_google/      # OCR text outputs (one .txt per document) + summary.csv
├── results/                        # Categorization output (jfk_categorization_55missing.csv) + checkpoint
│
└── manual/                         # Ground truth annotations, sample PDFs, and document index
    ├── jfk_files_index_manual.xlsx
    ├── pdf files/                  # Sample PDFs used for OCR evaluation
    └── Fwd_ JFK documents/         # Manually transcribed text files
```

## OCR Script (`ocr_missing_google.py`)

Reads `missing_file_ids.txt`, converts each PDF's pages to optimized JPEGs, and submits them to the Google Cloud Vision API with retry/backoff and checkpointing.

> **Note:** Google Cloud credentials are not included. Set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON file before running.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
python ocr_missing_google.py
```

You also need [poppler](https://poppler.freedesktop.org/) installed for `pdf2image`.

Output is written to `ocr_missing_output_google/`. If interrupted, rerunning resumes from the last checkpoint.

## License & Attribution

Provided as-is for academic research into JFK records.
