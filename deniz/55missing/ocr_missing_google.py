import io
import csv
import json
import time
import traceback
from pathlib import Path
from PIL import Image
from google.cloud import vision
from pdf2image import convert_from_path

BASE_DIR = Path.home() / "Desktop" / "Thesis"
PDF_DIR = BASE_DIR / "jfk_pdfs"
MISSING_FILE = BASE_DIR / "missing_file_ids.txt"
OUT_DIR = BASE_DIR / "ocr_missing_output_google"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = OUT_DIR / "progress.json"

# Retry configuration
MAX_RETRIES = 8
BASE_BACKOFF = 1.0  # Start with 1 second
MAX_BACKOFF = 60.0  # Cap at 60 seconds
PAGE_TIMEOUT = 45  # Seconds for per-page Vision API calls
PAGE_SLEEP = 0.3  # Sleep between page requests (seconds)

# Image optimization
PDF_DPI = 200  # Reduced from 300 to reduce processing time and payload
JPEG_QUALITY = 75  # JPEG compression quality
MAX_LONGEST_SIDE = 2000  # Optional: resize longest dimension if needed

# Read 55 IDs
ids = [line.strip() for line in MISSING_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"[INFO] Found {len(ids)} IDs to process")

client = vision.ImageAnnotatorClient()

def is_retryable_error(error_msg: str) -> bool:
    """Check if an error is retryable (transient)"""
    error_lower = str(error_msg).lower()
    retryable_patterns = [
        "503",  # Service unavailable
        "429",  # Rate limit
        "408",  # Request timeout
        "broken pipe",
        "connection reset",
        "connection timeout",
        "timeout",
        "temporarily unavailable",
        "deadline exceeded",
        "unavailable",
        "sendmsg",
        "recvmsg",
        "operation timed out",
    ]
    return any(pattern in error_lower for pattern in retryable_patterns)

def optimize_image_for_vision(pil_img):
    """
    Optimize PIL image for Google Vision API:
    - Save as JPEG with compression
    - Optionally resize if too large
    - Return bytes
    """
    # Check if resizing is needed (longest side > MAX_LONGEST_SIDE)
    width, height = pil_img.size
    longest = max(width, height)
    
    if longest > MAX_LONGEST_SIDE:
        ratio = MAX_LONGEST_SIDE / longest
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to JPEG for smaller payload
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    buf.seek(0)
    return buf.getvalue()

def ocr_pil_image_with_retry(pil_img, page_num: int = None, max_retries: int = MAX_RETRIES):
    """
    Extract text from a PIL image using Google Cloud Vision with retry logic and timeout.
    Returns: (text, error_msg, retry_count) where error_msg is None if successful
    """
    retry_count = 0
    
    for attempt in range(1, max_retries + 1):
        try:
            # Optimize image for Vision API
            image_bytes = optimize_image_for_vision(pil_img)
            image = vision.Image(content=image_bytes)
            
            # Call Vision API with timeout
            from google.api_core.gapic_v1 import client_info as grpc_client_info
            resp = client.document_text_detection(image=image, timeout=PAGE_TIMEOUT)
            
            if resp.error and resp.error.message:
                raise RuntimeError(f"Vision API error: {resp.error.message}")
            
            text = resp.full_text_annotation.text if resp.full_text_annotation else ""
            return text, None, retry_count
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            is_retryable = is_retryable_error(error_msg)
            
            if not is_retryable or attempt >= max_retries:
                # Not retryable or max retries exceeded
                return None, error_msg, retry_count
            
            # Exponential backoff
            wait_time = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
            # Add jitter (±10%)
            wait_time *= (0.9 + 0.2 * (attempt % 2))
            
            page_info = f" (page {page_num})" if page_num else ""
            print(f"       ⚠ Attempt {attempt}/{max_retries} failed{page_info}: {error_msg[:60]}")
            print(f"       ⏳ Retrying in {wait_time:.1f}s...")
            
            time.sleep(wait_time)
    
    return None, f"Failed after {max_retries} retries", retry_count

# Track results
results = []
errors_list = []
skipped = 0
successful = 0
failed = 0

# Load progress checkpoint
progress = {}
if PROGRESS_FILE.exists():
    try:
        progress = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        print(f"[INFO] Loaded progress checkpoint with {len(progress)} documents")
    except Exception as e:
        print(f"[WARNING] Could not load progress: {e}")
        progress = {}

print("\n[PROGRESS] Starting text extraction...\n")

for idx, doc_id in enumerate(ids, 1):
    pdf_path = PDF_DIR / f"{doc_id}.pdf"
    out_path = OUT_DIR / f"{doc_id}.txt"
    err_path = OUT_DIR / f"{doc_id}.error.txt"
    
    pdf_exists = pdf_path.exists()
    pages = 0
    error_msg = None

    # Check if output already exists and is non-empty (completely done)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[{idx:2d}/{len(ids)}] SKIP   {doc_id} (output exists)")
        skipped += 1
        results.append({
            "id": doc_id,
            "pdf_exists": "yes" if pdf_exists else "no",
            "pages": pages,
            "output_txt_path": str(out_path),
            "error": "SKIPPED (already processed)"
        })
        continue

    if not pdf_exists:
        print(f"[{idx:2d}/{len(ids)}] ERROR  {doc_id} (PDF not found)")
        error_msg = "PDF not found"
        failed += 1
        err_path.write_text(error_msg, encoding="utf-8")
        results.append({
            "id": doc_id,
            "pdf_exists": "no",
            "pages": 0,
            "output_txt_path": str(out_path),
            "error": error_msg
        })
        errors_list.append((doc_id, error_msg))
        continue

    try:
        # Convert all pages to images (single conversion) with optimized DPI
        page_images = convert_from_path(str(pdf_path), dpi=PDF_DPI)
        pages = len(page_images)
        
        # Determine resume point
        start_page = 1
        texts = []
        doc_stats = {
            "total_retries": 0,
            "failed_pages": []
        }
        
        if doc_id in progress:
            checkpoint = progress[doc_id]
            start_page = checkpoint.get("last_page", 0) + 1
            doc_stats = checkpoint.get("stats", {"total_retries": 0, "failed_pages": []})
            # Load previously extracted text
            if out_path.exists():
                existing = out_path.read_text(encoding="utf-8")
                # Split by the page separator (double newline between pages)
                texts = existing.split("\n\n")
                print(f"[{idx:2d}/{len(ids)}] RESUME {doc_id} from page {start_page}/{pages} ({len(texts)} pages loaded)")
            else:
                print(f"[{idx:2d}/{len(ids)}] START  {doc_id} ({pages} pages)")
        else:
            print(f"[{idx:2d}/{len(ids)}] START  {doc_id} ({pages} pages)")
        
        # Process remaining pages
        for page_num in range(start_page, pages + 1):
            page_img = page_images[page_num - 1]  # 0-indexed
            
            text, error, retry_count = ocr_pil_image_with_retry(page_img, page_num=page_num)
            doc_stats["total_retries"] += retry_count
            
            if error:
                # Page failed - add error placeholder
                error_placeholder = f"[ERROR page {page_num}: {error[:80]}]"
                texts.append(error_placeholder)
                doc_stats["failed_pages"].append(page_num)
                print(f"       ✗ Page {page_num}/{pages} FAILED: {error[:50]}")
            else:
                texts.append(text)
                # Print progress every 10 pages to reduce spam
                if page_num % 10 == 0 or page_num == pages:
                    print(f"       └─ Page {page_num}/{pages} processed")
            
            # Small sleep between requests to avoid rate limiting
            if page_num < pages:  # Don't sleep after last page
                time.sleep(PAGE_SLEEP)
            
            # Save checkpoint after each page
            progress[doc_id] = {
                "last_page": page_num,
                "total_pages": pages,
                "status": "in_progress",
                "stats": doc_stats
            }
            PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        
        # Write extracted text
        full_text = "\n\n".join(texts)
        out_path.write_text(full_text, encoding="utf-8")
        
        # Mark as complete in progress
        progress[doc_id]["status"] = "completed"
        PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        
        # Print document statistics
        page_errors = len(doc_stats["failed_pages"])
        status_symbol = "✓" if page_errors == 0 else "⚠"
        print(f"       {status_symbol} {doc_id} completed")
        print(f"          Pages: {pages} | Retries: {doc_stats['total_retries']} | Failed pages: {page_errors}")
        if doc_stats["failed_pages"]:
            print(f"          Failed pages: {doc_stats['failed_pages']}")
        print(f"          Output size: {len(full_text)} characters")
        
        successful += 1
        error_msg = f"{page_errors} pages failed" if page_errors > 0 else None
        results.append({
            "id": doc_id,
            "pdf_exists": "yes",
            "pages": pages,
            "output_txt_path": str(out_path),
            "error": error_msg if error_msg else ""
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"[{idx:2d}/{len(ids)}] ERROR  {doc_id} ({error_msg[:50]}...)")
        failed += 1
        err_path.write_text(f"{error_msg}\n\n{traceback.format_exc()}", encoding="utf-8")
        results.append({
            "id": doc_id,
            "pdf_exists": "yes",
            "pages": pages,
            "output_txt_path": str(out_path),
            "error": error_msg
        })
        errors_list.append((doc_id, error_msg))

# Write summary CSV
csv_path = OUT_DIR / "summary.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "pdf_exists", "pages", "output_txt_path", "error"])
    writer.writeheader()
    writer.writerows(results)

# Clean up progress for completed documents
completed_ids = {r["id"] for r in results if not r["error"] or r["error"] == "SKIPPED (already processed)"}
progress = {k: v for k, v in progress.items() if k not in completed_ids}
if progress:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")
else:
    # All done - remove progress file
    PROGRESS_FILE.unlink(missing_ok=True)
    print("[INFO] All documents completed - progress file removed")

# Print final summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Total IDs processed:        {len(ids)}")
print(f"Successful extractions:     {successful}")
print(f"Failed extractions:         {failed}")
print(f"Skipped (already done):     {skipped}")
print(f"\nSummary CSV:                {csv_path}")
print(f"Output folder:              {OUT_DIR}")
if progress:
    print(f"Incomplete documents:       {len(progress)} (in progress.json)")
print("="*70)

if errors_list:
    print(f"\nFIRST 5 ERRORS:")
    print("-"*70)
    for doc_id, msg in errors_list[:5]:
        print(f"\n{doc_id}:")
        print(f"  {msg[:100]}")
    print("-"*70)
