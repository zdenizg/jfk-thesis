import os
import base64
import json
import sys 
import pandas as pd
from pathlib import Path
from groq import Groq
from typing import List, Dict
import time
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import io
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

class DocumentCategorizer:
    def __init__(
        self, 
        api_key: str = None, 
        vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        text_model: str = "llama-3.1-8b-instant",
        output_prefix: str = "jfk_categorization",
        batch_size: int = 100
    ):
        """
        Initialize the categorizer with Groq API key.
        
        Args:
            api_key: Groq API key (if None, loads from GROQ_API_KEY env variable)
            vision_model: Groq vision model (default: llama-4-scout-17b-16e-instruct)
            text_model: Groq text model for categorization (default: llama-3.1-8b-instant)
            checkpoint_file: File to save processing progress
            batch_size: Save results every N pages
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key must be provided or set in GROQ_API_KEY environment variable")
        
        self.client = Groq(api_key=self.api_key)
        self.vision_model = vision_model
        self.text_model = text_model
        self.results = []
        self.output_prefix = output_prefix
        self.checkpoint_file = f"{output_prefix}_checkpoint.json"
        self.batch_size = batch_size
        self.processed_pages = self._load_checkpoint()
        
        print(f"🦙 Using Groq models:")
        print(f"   Vision: {vision_model} (594 TPS)")
        print(f"   Text: {text_model} (840 TPS)")
        print(f"📊 Checkpoint file: {self.checkpoint_file}")
        print(f"💾 Auto-saving every {batch_size} pages")
        if self.processed_pages:
            print(f"✅ Resuming: {len(self.processed_pages)} pages already processed")
    
    def _load_checkpoint(self) -> set:
        """Load previously processed pages from checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    return set(tuple(item) for item in data.get('processed_pages', []))
            except Exception as e:
                print(f"⚠️  Error loading checkpoint: {e}")
                return set()
        return set()
    
    def _save_checkpoint(self, filename: str, page_number: int):
        """Save current progress to checkpoint file."""
        self.processed_pages.add((filename, page_number))
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump({
                    'processed_pages': list(self.processed_pages),
                    'total_processed': len(self.processed_pages),
                    'last_updated': datetime.now().isoformat()
                }, f)
        except Exception as e:
            print(f"⚠️  Error saving checkpoint: {e}")
    
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
    
    def process_document(self, image_path: str, page_number: int = 1) -> Dict:
        """Process a single document page through combined analysis."""
        filename = os.path.basename(image_path)
        
        # Check if already processed
        if (filename, page_number) in self.processed_pages:
            # Don't print for every skipped page to reduce noise
            return None
        
        print(f"📄 {filename} p{page_number}", end=" ")
        
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
            "primary_characteristics": ", ".join(analysis.get("primary_characteristics", []))
        }
        
        # Save checkpoint
        self._save_checkpoint(filename, page_number)
        
        return result
    
    def process_directory(
        self, 
        directory_path: str, 
        file_extensions: List[str] = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff'],
        delay: float = 0.3,
        auto_save_csv: str = None
    ):
        """
        Process all documents in a directory with checkpointing and auto-save.
        
        Args:
            directory_path: Path to directory containing documents
            file_extensions: List of file extensions to process
            delay: Delay in seconds between processing pages (0.3s recommended for Groq)
            auto_save_csv: Path to auto-save CSV (saves every batch_size pages)
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        image_files = []
        
        for ext in file_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        image_files = sorted(set(image_files))
        
        # Count total pages
        total_pages = 0
        for img_file in image_files:
            total_pages += self.get_page_count(str(img_file))
        
        already_processed = len(self.processed_pages)
        remaining_pages = total_pages - already_processed
        
        print(f"\n📚 Found {len(image_files)} documents ({total_pages} total pages)")
        print(f"✅ Already processed: {already_processed} pages")
        print(f"📝 Remaining: {remaining_pages} pages")
        print(f"💰 Estimated cost: ${remaining_pages * 0.0003:.2f}")
        print(f"⏱️  Estimated time: {remaining_pages / 10 / 60:.1f} hours (~10 pages/min)\n")
        
        if not image_files:
            print("⚠️  No documents found in the specified directory")
            return
        
        total_pages_processed = 0
        start_time = time.time()
        
        for idx, image_path in enumerate(image_files, 1):
            image_path_str = str(image_path)
            page_count = self.get_page_count(image_path_str)
            filename = os.path.basename(image_path_str)
            
            # Check how many pages already processed for this file
            pages_done = sum(1 for p in self.processed_pages if p[0] == filename)
            
            if pages_done == page_count:
                print(f"[{idx}/{len(image_files)}] ⏭️  {filename} (all {page_count} pages done)")
                continue
            
            print(f"\n[{idx}/{len(image_files)}] 📖 {filename} ({page_count} pages, {pages_done} done)")
            
            for page_num in range(1, page_count + 1):
                result = self.process_document(image_path_str, page_num)
                if result:
                    self.results.append(result)
                    total_pages_processed += 1
                    
                    # Auto-save every batch_size pages
                    if auto_save_csv and total_pages_processed % self.batch_size == 0:
                        self._auto_save(auto_save_csv, total_pages_processed, start_time, remaining_pages)
                
                # Delay to respect rate limits
                time.sleep(delay)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed! Processed {len(self.results)} new pages in {elapsed/60:.1f} minutes")
        print(f"⚡ Average: {len(self.results)/(elapsed/60):.1f} pages/minute")
    
    def _auto_save(self, csv_path: str, pages_processed: int, start_time: float, total_remaining: int):
        """Auto-save results to CSV with progress stats."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(csv_path, index=False)
            
            elapsed = time.time() - start_time
            rate = pages_processed / (elapsed / 60) if elapsed > 0 else 0
            eta_minutes = (total_remaining - pages_processed) / rate if rate > 0 else 0
            
            print(f"\n💾 Auto-saved: {pages_processed} pages | {rate:.1f} pg/min | ETA: {eta_minutes:.0f}min")
    
    def save_results(self, output_path: str = None) -> pd.DataFrame:
        """Save results to CSV file and return DataFrame."""
        if output_path is None:
            output_path = f"{self.output_prefix}_final.csv"
        
        if not self.results:
            print("⚠️  No new results to save")
            return None
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics of the categorization results."""
        print("\n" + "="*60)
        print("📊 SUMMARY STATISTICS")
        print("="*60)
        print(f"Total document pages: {len(df)}")
        
        print(f"\n📁 Documents by type (top 10):")
        type_counts = df['document_type'].value_counts().head(10)
        for doc_type, count in type_counts.items():
            print(f"  • {doc_type}: {count}")
        
        print(f"\n🔍 OCR Difficulty:")
        difficulty_counts = df['ocr_difficulty'].value_counts()
        for difficulty, count in difficulty_counts.items():
            pct = count/len(df)*100
            print(f"  • {difficulty.capitalize()}: {count} ({pct:.1f}%)")
        
        print(f"\n📝 Document characteristics:")
        chars = {
            'Handwriting': df['includes_handwriting'].sum(),
            'Shadowy background': df['has_shadowy_background'].sum(),
            'Stamps': df['has_stamps'].sum(),
            'Redactions': df['has_redactions'].sum(),
            'Forms': df['has_forms'].sum(),
            'Tables': df['has_tables'].sum()
        }
        for char, count in chars.items():
            pct = count/len(df)*100
            print(f"  • {char}: {count} ({pct:.1f}%)")
        
        print(f"\n📋 Document quality:")
        quality_counts = df['document_quality'].value_counts()
        for quality, count in quality_counts.items():
            pct = count/len(df)*100
            print(f"  • {quality.capitalize()}: {count} ({pct:.1f}%)")
        
        print("="*60)
    
    def merge_with_existing(self, existing_csv: str, output_csv: str):
        """Merge current results with existing CSV file."""
        if not os.path.exists(existing_csv):
            print(f"⚠️  Existing file not found: {existing_csv}")
            return
        
        try:
            existing_df = pd.read_csv(existing_csv)
            new_df = pd.DataFrame(self.results)
            
            # Merge and remove duplicates based on filename + page_number
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=['filename', 'page_number'], keep='last')
            
            merged_df.to_csv(output_csv, index=False)
            print(f"✅ Merged {len(existing_df)} existing + {len(new_df)} new = {len(merged_df)} total pages")
            print(f"💾 Saved to: {output_csv}")
            
            return merged_df
        except Exception as e:
            print(f"❌ Error merging files: {e}")


# Example usage
if __name__ == "__main__":
    folder_num = sys.argv[1] if len(sys.argv) > 1 else "2"
    
    # Clean folder name for output files (replace hyphens with underscores)
    folder_clean = folder_num.replace("-", "_")
    
    DOCUMENTS_DIR = f"/Users/furkandemir/Desktop/Thesis/files/{folder_num}"
    output_prefix = f"jfk_categorization_{folder_clean}"
    
    print(f"\n📁 Processing folder: {folder_num}")
    print(f"📝 Output prefix: {output_prefix}\n")
    
    categorizer = DocumentCategorizer(
        vision_model="meta-llama/llama-4-scout-17b-16e-instruct",
        text_model="llama-3.1-8b-instant",
        output_prefix=output_prefix,
        batch_size=100
    )
    
    try:
        categorizer.process_directory(DOCUMENTS_DIR, delay=0.3)
        categorizer.save_results()
        print(f"\n✅ Saved: {output_prefix}_final.csv")
    except KeyboardInterrupt:
        categorizer.save_results(f"{output_prefix}_interrupted.csv")