# OCR Comparison Pipeline - Complete Structure

## 📁 Directory Structure

```
Thesis/scrapers/
├── .env                           # API keys (GROQ_API_KEY, HUGGINGFACE_API_KEY)
├── original_pdfs/                 # Source PDFs to process (already sampled)
│   ├── file1.pdf
│   ├── file2.pdf
│   └── ...
├── manual_extraction/             # Ground truth text files
│   ├── file1.txt
│   ├── file2.txt
│   └── ...
├── ocr_extraction/                # Output directory (auto-created)
│   ├── tesseract_only/
│   │   ├── file1.txt
│   │   └── file2.txt
│   ├── tesseract_preprocessed/
│   ├── deepseek_only/
│   ├── deepseek_preprocessed/
│   ├── tesseract_llama/
│   ├── tesseract_preprocessed_llama/
│   ├── deepseek_llama/
│   ├── deepseek_preprocessed_llama/
│   ├── progress.json              # Resume checkpoint
│   ├── results_summary.json       # Final metrics
│   └── comparison_report.csv      # Summary table
└── ocr_comparison.py              # Main script
```

## 🔧 Configuration

### API Keys (.env file)
```bash
GROQ_API_KEY=your-groq-api-key
HUGGINGFACE_API_KEY=your-huggingface-token
```

### Processing Parameters
- **DPI**: 400 (PDF to image conversion)
- **Tesseract Path**: `/opt/homebrew/bin/tesseract` (macOS)
- **Llama Model**: `llama-3.1-8b-instant`
- **Llama Temperature**: 0.1 (conservative)
- **Llama Max Tokens**: 8000

### Llama Cleanup Prompt
```xml
<role>
You are an expert OCR text cleanup specialist trained in restoring historical documents with high fidelity.
</role>
<task>
Your task is to correct OCR-generated errors in historical texts while preserving the original structure and meaning.
</task>
<instructions>
* Correct OCR spelling errors (e.g., "belie.vbe" → "believe", "gove.rnment" → "government")
* Remove OCR artifacts such as stray punctuation, broken words, and random characters
* Fix spacing issues (e.g., "th is" → "this", "document.The" → "document. The")
* Maintain all factual content exactly as-is — do not invent, omit, or alter meaning
* Preserve the original layout and formatting
* Do not include any notes, comments, or explanations
* Output only the cleaned version of the text
</instructions>
<output-format>
Plain text: return only the corrected version, no additional formatting or commentary.
</output-format>
<user-input>
INPUT TEXT:
{ocr_text}
</user-input>
```

## 🔬 OCR Methods (8 Total)

| # | Method Name | Image Preprocessing | OCR Engine | Post-Processing |
|---|-------------|---------------------|------------|-----------------|
| 1 | tesseract_only | ❌ None | Tesseract | ❌ None |
| 2 | tesseract_preprocessed | ✅ Level 2 | Tesseract | ❌ None |
| 3 | deepseek_only | ❌ None | DeepSeek OCR | ❌ None |
| 4 | deepseek_preprocessed | ✅ Level 2 | DeepSeek OCR | ❌ None |
| 5 | tesseract_llama | ❌ None | Tesseract | ✅ Llama Cleanup |
| 6 | tesseract_preprocessed_llama | ✅ Level 2 | Tesseract | ✅ Llama Cleanup |
| 7 | deepseek_llama | ❌ None | DeepSeek OCR | ✅ Llama Cleanup |
| 8 | deepseek_preprocessed_llama | ✅ Level 2 | DeepSeek OCR | ✅ Llama Cleanup |

## 🖼️ Level 2 Preprocessing Pipeline

**Applied only to methods with "preprocessed" in name**

```
Step 1: Convert to Grayscale
   ↓
Step 2: Denoise (fastNlMeansDenoising)
   ↓
Step 3: Contrast Enhancement (CLAHE)
   ↓
Step 4: Adaptive Thresholding (Binary conversion)
   ↓
Enhanced Image → OCR
```

**Preprocessing Parameters:**
- Denoise strength: h=10
- CLAHE clip limit: 2.0
- CLAHE grid size: 8x8
- Adaptive threshold: Gaussian, block size=11, C=2

## 🔄 Processing Flow

### Per Method Processing
```
For each method (tesseract_only, tesseract_preprocessed, etc.):
    
    1. Load progress.json (check what's already done)
    
    2. Create method output directory
    
    3. For each PDF in original_pdfs/:
        
        a. Check if already processed (resume capability)
        
        b. Convert PDF → Images (400 DPI)
        
        c. If "preprocessed" in method_name:
           Apply Level 2 preprocessing to each image
        
        d. If "tesseract" in method_name:
           Run Tesseract OCR
           Else if "deepseek" in method_name:
           Run DeepSeek OCR (HuggingFace API)
        
        e. Combine all pages into single text
        
        f. If "llama" in method_name:
           Send to Llama for cleanup via Groq API
        
        g. Save to ocr_extraction/{method_name}/{filename}.txt
        
        h. Update progress.json
        
        i. Update progress bar
    
    4. Calculate accuracy metrics for this method
    
    5. Save method results to results_summary.json
```

## 📊 Accuracy Metrics

**Calculated by comparing against manual_extraction/**

For each file and method:
- **Character Similarity**: Character-level matching (%)
- **Word Similarity**: Word-level matching (%)
- **Line Similarity**: Line-level matching (%)
- **Levenshtein Similarity**: Normalized edit distance (%)
- **Average Similarity**: Mean of above metrics (%)

## 📤 Output Files

### Per Method
```
ocr_extraction/{method_name}/{filename}.txt
```
- Raw extracted text (or Llama-cleaned text)
- Same filename as original PDF
- UTF-8 encoding

### Progress Tracking
```json
ocr_extraction/progress.json
{
  "tesseract_only": {
    "completed": ["file1.pdf", "file2.pdf"],
    "total": 25,
    "last_updated": "2024-11-25T14:30:00"
  },
  "tesseract_preprocessed": {
    "completed": ["file1.pdf"],
    "total": 25,
    "last_updated": "2024-11-25T14:35:00"
  }
}
```

### Results Summary
```json
ocr_extraction/results_summary.json
{
  "metadata": {
    "total_files": 25,
    "processing_date": "2024-11-25",
    "dpi": 400,
    "llama_model": "llama-3.1-8b-instant"
  },
  "methods": {
    "tesseract_only": {
      "average_accuracy": 87.5,
      "min_accuracy": 65.2,
      "max_accuracy": 98.1,
      "processing_time_seconds": 450
    }
  },
  "detailed_results": [
    {
      "filename": "file1.pdf",
      "methods": {
        "tesseract_only": {
          "character_similarity": 89.5,
          "word_similarity": 85.2,
          "line_similarity": 82.1,
          "levenshtein_similarity": 88.3,
          "average_similarity": 86.3
        }
      }
    }
  ]
}
```

### Comparison Report CSV
```csv
filename,tesseract_only,tesseract_preprocessed,deepseek_only,deepseek_preprocessed,tesseract_llama,tesseract_preprocessed_llama,deepseek_llama,deepseek_preprocessed_llama
file1.pdf,86.3,89.1,91.2,93.5,88.7,91.4,92.8,94.2
file2.pdf,78.5,82.3,85.6,88.9,81.2,84.7,87.1,90.3
...
```

## 🔄 Resume Capability

**How it works:**
1. Before processing each file, check `progress.json`
2. If file already processed for this method → skip
3. If script interrupted → restart and continue from last checkpoint
4. Progress updated after each file completion

**Manual reset:**
```bash
# Delete progress for specific method
# Edit progress.json and remove method entry

# Full reset
rm ocr_extraction/progress.json
```

## 📊 Progress Display

**During processing:**
```
Processing Method: tesseract_preprocessed (2/8)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 15/25 [60%] ETA: 5m 30s
Current: file015.pdf | Page 3/8 | Status: OCR in progress
```

**After completion:**
```
✓ tesseract_preprocessed complete
  Average Accuracy: 89.3%
  Processing Time: 12m 45s
  Files: 25/25
```

## 🛠️ Error Handling

**File-level errors:**
- If single file fails → log error, continue to next file
- Save error info to `errors.json`
- Don't stop entire pipeline

**API errors:**
- Retry with exponential backoff (3 attempts)
- If HuggingFace/Groq fails → log error, skip that file
- Continue with remaining files

**Resume after crash:**
- Check `progress.json` on restart
- Continue from last successful file

## 📦 Dependencies

```
PyPDF2
pytesseract
pdf2image
Pillow
opencv-python (cv2)
numpy
groq
requests
python-Levenshtein
pandas
python-dotenv
tqdm (progress bars)
```

## 🚀 Execution Flow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up .env file with API keys

# 3. Run the pipeline
python ocr_comparison.py

# The script will:
# - Process all 8 methods sequentially
# - Show progress for each method
# - Save results after each method completes
# - Generate comparison report at the end
```

## 📈 Expected Timeline

**Assumptions:**
- 25 PDF files
- Average 5 pages per PDF
- 400 DPI conversion

**Time estimates per method:**
- **tesseract_only**: ~8-10 minutes
- **tesseract_preprocessed**: ~12-15 minutes (preprocessing adds time)
- **deepseek_only**: ~10-12 minutes (API calls)
- **deepseek_preprocessed**: ~15-18 minutes
- **tesseract_llama**: ~15-20 minutes (Llama API adds time)
- **tesseract_preprocessed_llama**: ~20-25 minutes
- **deepseek_llama**: ~18-22 minutes
- **deepseek_preprocessed_llama**: ~25-30 minutes

**Total estimated time: 2.5 - 3.5 hours** for all 8 methods on 25 files

## 🎯 Success Criteria

**Pipeline is successful when:**
- ✅ All PDF files matched with manual extraction files
- ✅ All 8 methods complete without fatal errors
- ✅ Output files created in all method directories
- ✅ results_summary.json contains all accuracy metrics
- ✅ comparison_report.csv generated with all data

**Expected accuracy ranges (based on document quality):**
- **Good quality scans**: 85-95% baseline, 90-98% with preprocessing/Llama
- **Medium quality**: 70-85% baseline, 80-92% with preprocessing/Llama
- **Poor quality**: 50-70% baseline, 65-85% with preprocessing/Llama

---
