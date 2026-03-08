#!/usr/bin/env python3
"""
This script categorizes missing pages from documents in jfk-document-index.csv
where all_categorized = False. It only processes pages not already in 
jfk_categorization_combined.csv.
"""

import os
import base64
import json
import sys 
import pandas as pd
from pathlib import Path
from groq import Groq
from typing import List, Dict, Set, Tuple
import time
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import io
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()


class MissingPagesCategorizer:
    def __init__(
        self, 
        api_key: str = None, 
        vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        text_model: str = "llama-3.1-8b-instant",
        batch_size: int = 50
    ):
        """
        Initialize the categorizer with Groq API key.
        
        Args:
            api_key: Groq API key (if None, loads from GROQ_API_KEY env variable)
            vision_model: Groq vision model (default: llama-4-scout-17b-16e-instruct)
            text_model: Groq text model for categorization (default: llama-3.1-8b-instant)
            batch_size: Save results every N pages
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key must be provided or set in GROQ_API_KEY environment variable")
        
        self.client = Groq(api_key=self.api_key)
        self.vision_model = vision_model
        self.text_model = text_model
        self.results = []
        self.batch_size = batch_size
        self.pages_processed_this_session = 0
        
        # Define paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_csv_path = os.path.join(self.base_dir, "jfk-document-index.csv")
        self.combined_csv_path = os.path.join(self.base_dir, "results", "jfk_categorization_combined.csv")
        self.checkpoint_file = os.path.join(self.base_dir, "missing_pages_checkpoint.json")
        
        print(f"🦙 Using Groq models:")
        print(f"   Vision: {vision_model}")
        print(f"   Text: {text_model}")
        print(f"💾 Auto-saving every {batch_size} pages")
    
    def load_already_categorized_pages(self) -> Set[Tuple[str, int]]:
        """Load already categorized pages from the combined CSV."""
        if not os.path.exists(self.combined_csv_path):
            print(f"⚠️  Combined CSV not found: {self.combined_csv_path}")
            return set()
        
        df = pd.read_csv(self.combined_csv_path)
        categorized = set()
        for _, row in df.iterrows():
            categorized.add((row['filename'], int(row['page_number'])))
        
        print(f"✅ Loaded {len(categorized)} already categorized pages")
        return categorized
    
    def load_documents_needing_categorization(self) -> pd.DataFrame:
        """Load documents from index where all_categorized = False."""
        if not os.path.exists(self.index_csv_path):
            raise FileNotFoundError(f"Index CSV not found: {self.index_csv_path}")
        
        df = pd.read_csv(self.index_csv_path)
        # Filter for documents that are not fully categorized
        incomplete = df[df['all_categorized'] == False]
        print(f"📋 Found {len(incomplete)} documents needing categorization")
        return incomplete
    
    def get_page_count(self, image_path: str) -> int:
        """Return the number of pages/frames for the given file (PDF or multi-frame TIFF)."""
        file_ext = Path(image_path).suffix.lower()
        try:
            if file_ext == '.pdf':
                info = pdfinfo_from_path(image_path)
                return int(info.get("Pages", 1))
            elif file_ext in ['.tiff', '.tif']:
                with Image.open(image_path) as img:
                    return getattr(img, "n_frames", 1)
        except Exception as e:
            print(f"Error getting page count for {image_path}: {e}")
            return 1
        return 1

    def encode_image(self, image_path: str, page_number: int = 1, quality: int = 85) -> str:
        """
        Encode image to base64 string with compression.
        Handles PDFs and multi-frame TIFFs by converting the requested page/frame to an image.
        """
        file_ext = Path(image_path).suffix.lower()
        
        if file_ext == '.pdf':
            # Convert specific page of PDF to image
            try:
                images = convert_from_path(
                    image_path, 
                    first_page=page_number, 
                    last_page=page_number, 
                    dpi=150  # Balanced quality/cost
                )
                if images:
                    buffered = io.BytesIO()
                    images[0].save(buffered, format="JPEG", quality=quality, optimize=True)
                    return base64.b64encode(buffered.getvalue()).decode('utf-8')
                else:
                    raise Exception("No pages found in PDF for the requested page number")
            except Exception as e:
                print(f"Error converting PDF {image_path} page {page_number}: {e}")
                raise
        elif file_ext in ['.tiff', '.tif']:
            # Handle multi-frame TIFF
            try:
                with Image.open(image_path) as img:
                    frame_index = max(0, page_number - 1)
                    img.seek(frame_index)
                    buffered = io.BytesIO()
                    img_rgb = img.convert('RGB')
                    img_rgb.save(buffered, format="JPEG", quality=quality, optimize=True)
                    return base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Error converting TIFF {image_path} frame {page_number}: {e}")
                raise
        else:
            # Regular image file
            try:
                with Image.open(image_path) as img:
                    buffered = io.BytesIO()
                    img_rgb = img.convert('RGB')
                    img_rgb.save(buffered, format="JPEG", quality=quality, optimize=True)
                    return base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception:
                # Fallback to original file
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_and_categorize_document(self, image_path: str, page_number: int = 1) -> Dict:
        """
        Combined analysis and categorization in ONE API call to save costs.
        Returns complete analysis with category and OCR difficulty.
        """
        base64_image = self.encode_image(image_path, page_number)
        
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

        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": combined_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting
            if analysis_text.startswith("```"):
                analysis_text = analysis_text.split("```")[1]
                if analysis_text.startswith("json"):
                    analysis_text = analysis_text[4:]
                analysis_text = analysis_text.strip()
            
            return json.loads(analysis_text)
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for {image_path} page {page_number}: {e}")
            print(f"Response: {analysis_text[:200]}")
            return None
        except Exception as e:
            print(f"Error analyzing {image_path} page {page_number}: {e}")
            # Handle rate limiting
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print("⏸️  Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
            return None
    
    def process_page(self, image_path: str, filename: str, page_number: int) -> Dict:
        """Process a single document page through combined analysis."""
        print(f"  📄 Page {page_number}", end=" ")
        
        # Combined analysis
        analysis = self.analyze_and_categorize_document(image_path, page_number)
        if not analysis:
            print("❌")
            return None
        
        print("✓")
        
        # Combine results
        result = {
            "filename": filename,
            "page_number": page_number,
            "document_type": analysis.get("document_type", "unknown"),
            "ocr_difficulty": analysis.get("ocr_difficulty", "unknown"),
            "includes_handwriting": analysis.get("includes_handwriting", False),
            "has_shadowy_background": analysis.get("has_shadowy_background", False),
            "document_quality": analysis.get("document_quality", "unknown"),
            "text_density": analysis.get("text_density", "unknown"),
            "has_stamps": analysis.get("has_stamps", False),
            "has_redactions": analysis.get("has_redactions", False),
            "has_forms": analysis.get("has_forms", False),
            "has_tables": analysis.get("has_tables", False),
            "is_typewritten": analysis.get("is_typewritten", False),
            "paper_condition": analysis.get("paper_condition", "unknown"),
            "primary_characteristics": ", ".join(analysis.get("primary_characteristics", [])),
            "source_file": "missing_pages_categorization.csv"
        }
        
        return result
    
    def update_combined_csv(self):
        """Append new results to the combined CSV file."""
        if not self.results:
            return
        
        new_df = pd.DataFrame(self.results)
        
        if os.path.exists(self.combined_csv_path):
            existing_df = pd.read_csv(self.combined_csv_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates if any
            combined_df = combined_df.drop_duplicates(subset=['filename', 'page_number'], keep='last')
        else:
            combined_df = new_df
        
        combined_df.to_csv(self.combined_csv_path, index=False)
        print(f"💾 Updated combined CSV: {len(combined_df)} total pages")
    
    def update_index_csv(self):
        """Update the index CSV with new categorization counts."""
        # Reload combined CSV to get current counts
        if not os.path.exists(self.combined_csv_path):
            return
        
        combined_df = pd.read_csv(self.combined_csv_path)
        
        # Count occurrences of each filename
        from collections import Counter
        filename_counts = Counter(combined_df['filename'])
        
        # Load index CSV
        index_df = pd.read_csv(self.index_csv_path)
        
        # Update categorized column
        index_df['categorized'] = index_df['filename'].apply(lambda x: filename_counts.get(x, 0))
        
        # Update all_categorized column
        index_df['all_categorized'] = index_df['categorized'] >= index_df['number_of_pages']
        
        # Save updated index
        index_df.to_csv(self.index_csv_path, index=False)
        
        categorized_count = (index_df['all_categorized'] == True).sum()
        print(f"📊 Updated index CSV: {categorized_count}/{len(index_df)} documents fully categorized")
    
    def save_checkpoint(self, processed_files: Set[str]):
        """Save checkpoint of processed files."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump({
                    'processed_files': list(processed_files),
                    'total_processed': len(processed_files),
                    'last_updated': datetime.now().isoformat()
                }, f)
        except Exception as e:
            print(f"⚠️  Error saving checkpoint: {e}")
    
    def process_missing_pages(self, delay: float = 0.3, max_documents: int = None):
        """
        Main processing loop for missing pages.
        
        Args:
            delay: Delay in seconds between API calls
            max_documents: Maximum number of documents to process (None = all)
        """
        # Load already categorized pages
        categorized_pages = self.load_already_categorized_pages()
        
        # Load documents needing categorization
        incomplete_docs = self.load_documents_needing_categorization()
        
        if len(incomplete_docs) == 0:
            print("✅ All documents are fully categorized!")
            return
        
        # Calculate total missing pages
        total_missing = 0
        for _, row in incomplete_docs.iterrows():
            filename = row['filename']
            num_pages = row['number_of_pages']
            for page in range(1, num_pages + 1):
                if (filename, page) not in categorized_pages:
                    total_missing += 1
        
        print(f"\n📝 Total missing pages to categorize: {total_missing}")
        print(f"💰 Estimated cost: ${total_missing * 0.0003:.2f}")
        print(f"⏱️  Estimated time: {total_missing / 10 / 60:.1f} hours (~10 pages/min)\n")
        
        if max_documents:
            incomplete_docs = incomplete_docs.head(max_documents)
            print(f"⚠️  Limiting to first {max_documents} documents\n")
        
        start_time = time.time()
        documents_processed = 0
        
        for idx, (_, row) in enumerate(incomplete_docs.iterrows(), 1):
            filename = row['filename']
            file_location = row['file_location']
            num_pages = row['number_of_pages']
            
            if pd.isna(file_location) or not file_location:
                print(f"[{idx}/{len(incomplete_docs)}] ⚠️  {filename} - No file location, skipping")
                continue
            
            if not os.path.exists(file_location):
                print(f"[{idx}/{len(incomplete_docs)}] ⚠️  {filename} - File not found, skipping")
                continue
            
            # Find missing pages for this document
            missing_pages = []
            for page in range(1, num_pages + 1):
                if (filename, page) not in categorized_pages:
                    missing_pages.append(page)
            
            if not missing_pages:
                print(f"[{idx}/{len(incomplete_docs)}] ⏭️  {filename} - All pages already done")
                continue
            
            print(f"\n[{idx}/{len(incomplete_docs)}] 📖 {filename} ({len(missing_pages)} missing pages: {missing_pages[:5]}{'...' if len(missing_pages) > 5 else ''})")
            
            for page_num in missing_pages:
                result = self.process_page(file_location, filename, page_num)
                if result:
                    self.results.append(result)
                    self.pages_processed_this_session += 1
                    categorized_pages.add((filename, page_num))
                    
                    # Auto-save every batch_size pages
                    if self.pages_processed_this_session % self.batch_size == 0:
                        print(f"\n💾 Auto-save at {self.pages_processed_this_session} pages...")
                        self.update_combined_csv()
                        self.update_index_csv()
                        self.results = []  # Clear results after saving
                        
                        elapsed = time.time() - start_time
                        rate = self.pages_processed_this_session / (elapsed / 60) if elapsed > 0 else 0
                        pages_remaining = total_missing - self.pages_processed_this_session
                        eta_minutes = pages_remaining / rate if rate > 0 else 0
                        print(f"⚡ Rate: {rate:.1f} pg/min | Remaining: {pages_remaining} | ETA: {eta_minutes:.0f}min\n")
                
                # Delay to respect rate limits
                time.sleep(delay)
            
            documents_processed += 1
        
        # Final save
        if self.results:
            print(f"\n💾 Final save...")
            self.update_combined_csv()
            self.update_index_csv()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed! Processed {self.pages_processed_this_session} pages in {elapsed/60:.1f} minutes")
        if self.pages_processed_this_session > 0:
            print(f"⚡ Average: {self.pages_processed_this_session/(elapsed/60):.1f} pages/minute")


def main():
    # Parse command line arguments
    max_docs = None
    if len(sys.argv) > 1:
        try:
            max_docs = int(sys.argv[1])
            print(f"📝 Will process maximum {max_docs} documents")
        except ValueError:
            print(f"Usage: python {sys.argv[0]} [max_documents]")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("🔍 MISSING PAGES CATEGORIZER")
    print("="*60 + "\n")
    
    categorizer = MissingPagesCategorizer(
        vision_model="meta-llama/llama-4-scout-17b-16e-instruct",
        text_model="llama-3.1-8b-instant",
        batch_size=50
    )
    
    try:
        categorizer.process_missing_pages(delay=0.3, max_documents=max_docs)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Saving progress...")
        if categorizer.results:
            categorizer.update_combined_csv()
            categorizer.update_index_csv()
        print("✅ Progress saved. You can resume later.")


if __name__ == "__main__":
    main()
