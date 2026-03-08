# Topic Modeling on JFK Assassination Files (NLP Pipeline)

This repository contains the workflow and code used for **topic modeling on 2,566 recently declassified JFK assassination documents** (totaling **83,621 pages**). The project involves indexing, categorizing, OCR extraction, and tokenization of the dataset in preparation for NLP-based topic modeling.

---

## Thesis Topic

**Title:** *Topic Modeling and Thematic Analysis of JFK Assassination Files Using NLP*

---

## 📁 Folder Structure

```bash
Thesis/
.
├── README.md                          # This file
│
├── scrapers/                          # Document scraping tools
│   ├── README.md                     # Scraper documentation
│   ├── colab_ocr_comparison.ipynb    # Google Colab implementation
│   ├── ocr_comparison.py             # Main OCR comparison pipeline
│   ├── preprocess_scraper.py         # Document preprocessing
│   ├── test_scraper.py               # Scraper testing utilities
│   ├── tokenizer.py                  # Text tokenization tools
│   ├── requirements.txt              # Scraper dependencies
│   └── sample_files/                 # Sample documents for testing
│
├── organizers/                        # Document organization tools
│   ├── categorizer.py                # Document categorization
│   ├── combiner.py                   # Result combination utilities
│   ├── process_all.py                # Batch processing script
│   ├── jfk_categorization_101+_checkpoint.json
│   ├── requirements.txt              # Organizer dependencies
│   └── results/                      # Processing results
│
├── files/                            # Organized document storage
│   ├── 1/                           # 1-page documents
│   ├── 2/
│   ├── 3/
│   ├── 4/    
│   ├── 5/      
│   ├── 6-10/                        # 6-10 page documents
│   ├── 11-20/   
│   ├── 21-30/ 
│   ├── 31-40/
│   ├── 41-50/
│   ├── 51-60/
│   ├── 61-70/ 
│   ├── 71-80/
│   ├── 81-90/
│   ├── 91-100/
│   └── 101+/
│
├── archive/                         # Legacy scripts and trials
│   ├── 1_page_scraper.py            # Original scraper
│   ├── Blank Diagram.PDF            # Project diagram
│   ├── extracted_texts.xlsx
│   ├── index.py
│   ├── jfk_categorization_final.csv
│   ├── jfk_categorization_progress.csv
│   ├── trial.py      
│   └── trial2.py
│
├── final-tobedoneaftertests/
├── jfk_documents_index.xlsx         # Master document index
└── model-comparison.pdf             # OCR method comparison report

24 directories, 25 files

```

---

## Workflow Overview

### 1. Document Indexing

Run `index.py` to index all PDF files by ID and page count → **`jfk_documents_index.csv`**.

### 2. Folder Organization (via Terminal)

Group PDFs by page count into folders: `1`, `2`, `3`, ..., `101+`.

### 3. Document Categorization (LLM)

Run `categorizer.py` on **ALL documents** using **Groq Llama models** and **Leo Prompt Optimizer**.
Outputs → **`document_categorization.csv`**.

### 4. OCR Method Comparison & Selection

Run `ocr_comparison.py` on a **sample of ~20-30 documents** (stratified by page count and document category) to:
- Test 8 OCR methods: Tesseract (raw/preprocessed), DeepSeek (raw/preprocessed), with/without Llama cleanup
- Compare accuracy against manually extracted ground truth
- Generate **`comparison_report.csv`** with accuracy metrics
- **Select optimal OCR method** based on results

**Output:** 
- `comparison_report.csv` - Method performance comparison
- `results_summary.json` - Detailed accuracy metrics

### 5. High-Quality OCR + Preprocessing - Full Dataset

Run `ocr_extraction_final.py` using the **selected optimal methodology** from Step 4:
- Process all ~80,000 pages using best-performing OCR approach
- Output → **`extracted_texts.csv`**

### 6. Tokenization

Run `tokenizer.py` to tokenize extracted text → **`extracted_tokens.csv`**.

### 7. TF-IDF + Topic Modeling (Next Phase)

Planned implementation of **TF-IDF** followed by **LDA** or **BERTopic** for topic discovery.

---

## Tech Stack

* Python 3
* **Groq Llama models** (via `categorizer.py`)
* Leo Prompt Optimizer
* **Tesseract OCR**
* **DeepSeek-OCR** (with Transformers)
* **Google Colab** (A100/V100 GPU for OCR processing)
* Pandas, NumPy, Scikit-learn
* **python-Levenshtein** (for OCR accuracy measurement)
* TF-IDF, LDA, BERTopic (upcoming)

---

## Next Steps

1. Compute TF-IDF scores and identify top tokens per document.
2. Apply **LDA** or **BERTopic** for thematic clustering.
3. Visualize topic distributions across time and categories.
4. Evaluate semantic coherence and interpret patterns related to the JFK investigation.

---

## License

This project is released under the **MIT License**.
