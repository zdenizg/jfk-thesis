# JFK OCR Thesis Materials

This repository contains tools and results for extracting and analyzing text from JFK assassination documents released as part of the 2025 records release.

## Repository Structure

```
.
├── ocr_missing_google.py        # Google Cloud Vision OCR script for the 55 missing PDFs
├── missing_file_ids.txt         # List of 55 document IDs that were missing from the main dataset
├── ocr_missing_output_google/   # OCR output: one .txt per document + summary.csv
├── requirements.txt             # Python dependencies for the OCR script
│
├── manual/                      # Manually curated document index (shared project work)
│   └── jfk_files_index_manual.xlsx
│
└── deniz/                       # Personal work directory
    ├── Thesis_Proposal.pdf
    ├── 55missing/               # Extended analysis of the 55 missing documents
    └── manual/                  # Manual ground truth annotations + sample PDFs
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
