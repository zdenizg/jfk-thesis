# JFK OCR Thesis Materials

This repository contains tools and results for extracting and analyzing text from JFK assassination documents released as part of the 2025 records release.

## Repository Structure

```
.
├── ocr_missing_google.py             # Google Cloud Vision OCR script for the 55 missing PDFs
├── categorize_55_missing.py          # Classifies each page using Groq (document type, OCR
│                                     #   difficulty, handwriting, shadowy background, etc.)
├── check_missing_ids.py              # Checks which of the 55 PDFs are present in local jfk_pdfs/
├── missing_file_ids.txt              # List of 55 document IDs missing from the main dataset
├── requirements.txt                  # Python dependencies (google-cloud-vision, pdf2image, pillow)
├── Thesis_Proposal.pdf               # Thesis proposal document
│
├── ocr_missing_output_google/        # OCR outputs for the 55 documents
│   ├── <doc_id>.txt                  # Extracted text, one file per document (57 files)
│   ├── summary.csv                   # Per-document stats (pages, characters, processing time)
│   ├── missing_pages_categorization.csv  # Page-level categorization results
│   └── categorization_progress.sqlite   # SQLite checkpoint for categorization runs
│
├── results/
│   ├── jfk_categorization_55missing.csv  # Final categorization output (one row per page)
│   └── checkpoint_55missing.json         # Checkpoint to resume categorization if interrupted
│
└── manual/                           # Ground truth annotations and sample documents
    ├── jfk_files_index_manual.xlsx   # Manually recorded metadata (page count, type, difficulty, etc.)
    ├── pdf files/                    # 31 sample PDFs used for OCR evaluation
    ├── Fwd_ JFK documents/           # Manually transcribed text files for the sample documents
    └── Fwd_ JFK documents.zip        # Zipped archive of the above
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
