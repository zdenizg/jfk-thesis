"""
Google Vision OCR Scraper - Optimized Version
Extract text from PDFs using Google Cloud Vision API.
"""

import os
import time
import io
from pathlib import Path
from typing import List, Optional

# Image processing
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
from tqdm import tqdm
from google.cloud import vision

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    # Paths
    'original_pdfs_dir': '/Users/furkandemir/Desktop/Thesis/files/1/simple',
    
    # Output Directory (will be created)
    'ocr_output_dir': '/Users/furkandemir/Desktop/Thesis/ocr_output',

    # PDF Processing
    'dpi': 300,
    'image_format': 'PNG',
    
    # Processing settings
    'batch_size': 5,  # Process multiple pages before saving
    'rate_limit_delay': 0.2,  # Delay between API calls
    'max_retries': 3,  # Maximum retry attempts for failed OCR
    
    # Image enhancement
    'apply_preprocessing': True,  # Set to False if documents are already high quality
    'contrast_factor': 1.5,  # Increase contrast for faded documents
}

# Verify paths exist
if not Path(CONFIG['original_pdfs_dir']).exists():
    print(f"⚠️ WARNING: Input directory does not exist: {CONFIG['original_pdfs_dir']}")
    print("Check the path in CONFIG. If using 'Upload', ensure you ran the upload cell above and updated the path.")
else:
    print(f"✅ Input directory found: {CONFIG['original_pdfs_dir']}")


class GoogleOCRScraper:
    """
    Scraper to extract text from PDFs using Google Cloud Vision API
    """

    def __init__(self):
        """Initialize the OCR scraper"""

        # Setup paths
        self.original_pdfs_dir = Path(CONFIG['original_pdfs_dir'])
        self.ocr_output_dir = Path(CONFIG['ocr_output_dir'])
        self.ocr_output_dir.mkdir(exist_ok=True, parents=True)

        # Statistics
        self.stats = {
            'total_pages': 0,
            'successful_pages': 0,
            'failed_pages': 0,
            'total_api_calls': 0
        }

        # Initialize Google Vision Client
        self._init_client()

    def _init_client(self):
        """Initialize Google Cloud Vision Client"""
        try:
            self.client = vision.ImageAnnotatorClient()
            print("✅ Google Cloud Vision Client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Google Cloud Vision Client: {e}")
            print("Ensure you have uploaded your JSON key and set GOOGLE_APPLICATION_CREDENTIALS.")
            self.client = None

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better OCR results
        """
        if not CONFIG['apply_preprocessing']:
            return image
        
        try:
            # Convert to grayscale (often better for text recognition)
            if image.mode != 'L':
                image = image.convert('L')
            
            # Increase contrast for faded documents
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(CONFIG['contrast_factor'])
            
            return image
        except Exception as e:
            print(f"    Warning: Preprocessing failed, using original image: {e}")
            return image

    def ocr_google(self, image: Image.Image, page_num: int = 0) -> str:
        """
        Run Google Cloud Vision OCR on a PIL Image with retry logic
        """
        if not self.client:
            return "ERROR: Google Cloud Vision Client not initialized"

        for attempt in range(CONFIG['max_retries']):
            try:
                # Preprocess image
                processed_image = self.preprocess_image(image)
                
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                processed_image.save(img_byte_arr, format=CONFIG['image_format'])
                content = img_byte_arr.getvalue()

                # Prepare the image for Vision API
                vision_image = vision.Image(content=content)

                # Perform DOCUMENT text detection (better for documents than regular text detection)
                response = self.client.document_text_detection(image=vision_image)
                
                self.stats['total_api_calls'] += 1

                if response.error.message:
                    raise Exception(f'{response.error.message}')

                # Get the full text annotation
                if response.full_text_annotation:
                    return response.full_text_annotation.text
                else:
                    return ""

            except Exception as e:
                error_str = str(e)
                
                # Handle rate limiting
                if "Quota" in error_str or "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = (attempt + 1) * 5
                    print(f"    ⚠️ Rate limit hit on page {page_num}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # Handle other errors
                elif attempt < CONFIG['max_retries'] - 1:
                    print(f"    ⚠️ Error on page {page_num} (attempt {attempt + 1}/{CONFIG['max_retries']}): {error_str}")
                    time.sleep(2)
                    continue
                else:
                    print(f"    ❌ Failed after {CONFIG['max_retries']} attempts: {error_str}")
                    return f"ERROR: {error_str}"
        
        return "ERROR: Max retries exceeded"

    def process_pdf(self, pdf_path: Path) -> str:
        """
        Process a PDF file and extract text
        """
        # Convert PDF to images
        try:
            images = convert_from_path(str(pdf_path), dpi=CONFIG['dpi'])
            self.stats['total_pages'] += len(images)
        except Exception as e:
            print(f"  ❌ PDF conversion error for {pdf_path.name}: {e}")
            self.stats['failed_pages'] += 1
            return ""

        all_text_parts = []
        total_pages = len(images)

        # Process each page
        for page_num, image in enumerate(images, 1):
            try:
                page_text = self.ocr_google(image, page_num)
                
                if page_text and not page_text.startswith("ERROR"):
                    all_text_parts.append(page_text)
                    self.stats['successful_pages'] += 1
                    print(f"    ✓ Page {page_num}/{total_pages} done")
                else:
                    all_text_parts.append(f"[ERROR ON PAGE {page_num}: {page_text}]")
                    self.stats['failed_pages'] += 1
                    print(f"    ✗ Page {page_num}/{total_pages} failed")
                
                # Delay to avoid hitting rate limits
                time.sleep(CONFIG['rate_limit_delay'])

            except Exception as e:
                print(f"    ❌ Error processing page {page_num}: {e}")
                all_text_parts.append(f"[ERROR ON PAGE {page_num}]")
                self.stats['failed_pages'] += 1

        # Combine all pages
        full_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text_parts)
        return full_text

    def run_scraper(self):
        """Run the scraping process for all PDFs"""
        print("="*60)
        print("GOOGLE CLOUD VISION OCR SCRAPER - OPTIMIZED")
        print("="*60)
        print(f"Configuration:")
        print(f"  DPI: {CONFIG['dpi']}")
        print(f"  Image Format: {CONFIG['image_format']}")
        print(f"  Rate Limit Delay: {CONFIG['rate_limit_delay']}s")
        print(f"  Max Retries: {CONFIG['max_retries']}")
        print(f"  Preprocessing: {'Enabled' if CONFIG['apply_preprocessing'] else 'Disabled'}")
        print("="*60)

        # Get PDF files
        pdf_files = sorted(list(self.original_pdfs_dir.glob('*.pdf')))
        print(f"\nFound {len(pdf_files)} PDF files in {self.original_pdfs_dir}")

        start_time = time.time()

        for pdf_idx, pdf_file in enumerate(pdf_files, 1):
            output_file = self.ocr_output_dir / f"{pdf_file.stem}.txt"
            
            # Skip if already exists
            if output_file.exists():
                print(f"\n[{pdf_idx}/{len(pdf_files)}] ✓ Skipping {pdf_file.name} (already processed)")
                continue

            print(f"\n[{pdf_idx}/{len(pdf_files)}] Processing {pdf_file.name}...")
            pdf_start_time = time.time()
            
            text = self.process_pdf(pdf_file)
            
            # Save extracted text
            if text:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                elapsed = time.time() - pdf_start_time
                print(f"  ✅ Saved to {output_file.name} ({elapsed:.1f}s)")
            else:
                print(f"  ❌ Failed to extract text from {pdf_file.name}")
        
        # Print final statistics
        total_elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("SCRAPING COMPLETE!")
        print("="*60)
        print(f"Total time: {total_elapsed/60:.1f} minutes")
        print(f"Total pages processed: {self.stats['total_pages']}")
        print(f"Successful pages: {self.stats['successful_pages']}")
        print(f"Failed pages: {self.stats['failed_pages']}")
        print(f"Success rate: {self.stats['successful_pages']/self.stats['total_pages']*100:.1f}%" if self.stats['total_pages'] > 0 else "N/A")
        print(f"Total API calls: {self.stats['total_api_calls']}")
        print("="*60)


if __name__ == "__main__":
    scraper = GoogleOCRScraper()
    scraper.run_scraper()