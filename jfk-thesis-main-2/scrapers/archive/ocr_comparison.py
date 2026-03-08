"""
OCR Comparison Pipeline
Compare multiple OCR methods: Tesseract, DeepSeek OCR, with/without preprocessing, with/without Llama cleanup
"""

import os
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import difflib

# Image processing
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from groq import Groq

# Text comparison and data handling
import Levenshtein
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()


# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    # Paths
    'original_pdfs_dir': '/Users/furkandemir/Desktop/Thesis/scrapers/sample_files/original_pdfs',
    'manual_extraction_dir': '/Users/furkandemir/Desktop/Thesis/scrapers/sample_files/manual_extraction',
    'ocr_extraction_dir': '/Users/furkandemir/Desktop/Thesis/scrapers/sample_files/ocr_extraction',
    
    
    # Processing parameters
    'dpi': 400,
    'tesseract_path': '/opt/homebrew/bin/tesseract',
    
    # Llama settings
    'llama_model': 'llama-3.1-8b-instant',
    'llama_temperature': 0.1,
    'llama_max_tokens': 8000,
    'llama_cleanup_prompt': """

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

    """,
    
    # DeepSeek vLLM settings (replace existing deepseek settings)
    'use_deepseek_vllm': True,
    'deepseek_model': 'deepseek-ai/DeepSeek-OCR',
    'deepseek_prompt': '<image>\nFree OCR.',
    'deepseek_max_tokens': 8192,
    'deepseek_temperature': 0.0,
    
    # Preprocessing parameters
    'denoise_h': 10,
    'clahe_clip_limit': 2.0,
    'clahe_grid_size': (8, 8),
    'adaptive_block_size': 11,
    'adaptive_c': 2,
    
}


class OCRMethodComparator:
    """
    Compare multiple OCR methods with different preprocessing combinations
    """
    
    def __init__(self):
        """Initialize the OCR comparator"""
        
        # Setup paths
        self.original_pdfs_dir = Path(CONFIG['original_pdfs_dir'])
        self.manual_extraction_dir = Path(CONFIG['manual_extraction_dir'])
        self.ocr_extraction_dir = Path(CONFIG['ocr_extraction_dir'])
        self.ocr_extraction_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup Tesseract
        if CONFIG['tesseract_path']:
            pytesseract.pytesseract.tesseract_cmd = CONFIG['tesseract_path']

        # Load DeepSeek vLLM model
        if CONFIG.get('use_deepseek_vllm', False):
            self._load_deepseek_vllm()
        else:
            print("Warning: DeepSeek vLLM disabled. DeepSeek methods will fail.")
        
        # Setup API clients
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        
        if self.groq_api_key:
            self.groq_client = Groq(api_key=self.groq_api_key)
        else:
            print("Warning: GROQ_API_KEY not found. Llama methods will fail.")
        
        # Define OCR methods
        self.methods = [
            'tesseract_only',
            'tesseract_preprocessed',
            'deepseek_only',
            'deepseek_preprocessed',
            'tesseract_llama',
            'tesseract_preprocessed_llama',
            'deepseek_llama',
            'deepseek_preprocessed_llama',
        ]
        
        # Progress tracking
        self.progress_file = self.ocr_extraction_dir / 'progress.json'
        self.results_file = self.ocr_extraction_dir / 'results_summary.json'
        self.errors_file = self.ocr_extraction_dir / 'errors.json'
        
        self.progress_data = self.load_progress()
        self.results_data = self.load_results()
        self.errors_data = self.load_errors()
    
    def _load_deepseek_vllm(self):
        """Load DeepSeek-OCR model using vLLM"""
        try:
            print("Loading DeepSeek-OCR with vLLM (first time downloads ~6.7GB)...")
            
            # Create vLLM instance
            self.deepseek_llm = LLM(
                model=CONFIG['deepseek_model'],
                enable_prefix_caching=False,
                mm_processor_cache_gb=0,
                logits_processors=[NGramPerReqLogitsProcessor]
            )
            
            # Create sampling parameters (reusable)
            self.deepseek_sampling_params = SamplingParams(
                temperature=CONFIG['deepseek_temperature'],
                max_tokens=CONFIG['deepseek_max_tokens'],
                extra_args=dict(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822},
                ),
                skip_special_tokens=False,
            )
            
            print("  ✓ DeepSeek-OCR with vLLM loaded successfully")
            
        except ImportError as e:
            print(f"  ✗ vLLM not installed: {e}")
            print("  Install: pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly")
            self.deepseek_llm = None
            self.deepseek_sampling_params = None
        except Exception as e:
            print(f"  ✗ Error loading DeepSeek with vLLM: {e}")
            self.deepseek_llm = None
            self.deepseek_sampling_params = None

    # ============================================
    # PROGRESS AND STATE MANAGEMENT
    # ============================================
    
    def load_progress(self) -> Dict:
        """Load progress from JSON file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self):
        """Save progress to JSON file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress_data, f, indent=2)
    
    def load_results(self) -> Dict:
        """Load results from JSON file"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {
            'metadata': {
                'dpi': CONFIG['dpi'],
                'llama_model': CONFIG['llama_model'],
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'methods': {},
            'detailed_results': []
        }
    
    def save_results(self):
        """Save results to JSON file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results_data, f, indent=2)
    
    def load_errors(self) -> Dict:
        """Load errors from JSON file"""
        if self.errors_file.exists():
            with open(self.errors_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_errors(self):
        """Save errors to JSON file"""
        with open(self.errors_file, 'w') as f:
            json.dump(self.errors_data, f, indent=2)
    
    def update_progress(self, method: str, pdf_file: str, total: int):
        """Update progress for a method"""
        if method not in self.progress_data:
            self.progress_data[method] = {
                'completed': [],
                'total': total,
                'last_updated': None
            }
        
        if pdf_file not in self.progress_data[method]['completed']:
            self.progress_data[method]['completed'].append(pdf_file)
        
        self.progress_data[method]['last_updated'] = datetime.now().isoformat()
        self.save_progress()
    
    def is_processed(self, method: str, pdf_file: str) -> bool:
        """Check if a file has already been processed for a method"""
        if method not in self.progress_data:
            return False
        return pdf_file in self.progress_data[method]['completed']
    
    def log_error(self, method: str, pdf_file: str, error: str):
        """Log an error"""
        if method not in self.errors_data:
            self.errors_data[method] = {}
        
        self.errors_data[method][pdf_file] = {
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
        self.save_errors()
    
    # ============================================
    # IMAGE PREPROCESSING
    # ============================================
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply Level 2 preprocessing to improve OCR accuracy
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Denoise
        denoised = cv2.fastNlMeansDenoising(
            gray, 
            None, 
            h=CONFIG['denoise_h'], 
            templateWindowSize=7, 
            searchWindowSize=21
        )
        
        # Step 3: Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=CONFIG['clahe_clip_limit'], 
            tileGridSize=CONFIG['clahe_grid_size']
        )
        enhanced = clahe.apply(denoised)
        
        # Step 4: Adaptive thresholding (binarization)
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            CONFIG['adaptive_block_size'],
            CONFIG['adaptive_c']
        )
        
        # Convert back to PIL Image
        return Image.fromarray(binary)
    
    # ============================================
    # OCR METHODS
    # ============================================
    
    def ocr_tesseract(self, image: Image.Image) -> str:
        """
        Run Tesseract OCR on an image
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text
        """
        try:
            text = pytesseract.image_to_string(image, lang='eng')
            return text
        except Exception as e:
            print(f"Tesseract error: {e}")
            return ""
    
    def ocr_deepseek(self, image: Image.Image) -> str:
        """
        Run DeepSeek OCR using vLLM
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text
        """
        if not hasattr(self, 'deepseek_llm') or self.deepseek_llm is None:
            return "ERROR: DeepSeek vLLM not loaded"
        
        try:
            # Convert image to RGB (vLLM requires RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Prepare input for vLLM
            model_input = [{
                "prompt": CONFIG['deepseek_prompt'],
                "multi_modal_data": {"image": image}
            }]
            
            # Generate output
            model_outputs = self.deepseek_llm.generate(
                model_input,
                self.deepseek_sampling_params
            )
            
            # Extract text from output
            if model_outputs and len(model_outputs) > 0:
                return model_outputs[0].outputs[0].text
            else:
                return "ERROR: No output from DeepSeek"
        
        except Exception as e:
            print(f"DeepSeek vLLM error: {e}")
            return f"ERROR: {str(e)}"
    
    def cleanup_with_llama(self, text: str) -> str:
        """
        Cleanup OCR text using Llama via Groq
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not self.groq_api_key:
            return "ERROR: GROQ_API_KEY not set"
        
        try:
            prompt = CONFIG['llama_cleanup_prompt'].format(ocr_text=text)
            
            completion = self.groq_client.chat.completions.create(
                model=CONFIG['llama_model'],
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=CONFIG['llama_temperature'],
                max_tokens=CONFIG['llama_max_tokens']
            )
            
            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"Llama cleanup error: {e}")
            return f"ERROR: {str(e)}"
    
    # ============================================
    # PDF PROCESSING
    # ============================================
    
    def process_pdf_with_method(self, pdf_path: Path, method: str) -> str:
        """
        Process a PDF file with a specific OCR method
        
        Args:
            pdf_path: Path to PDF file
            method: Method name (e.g., 'tesseract_preprocessed_llama')
            
        Returns:
            Extracted and processed text
        """
        # Convert PDF to images
        try:
            images = convert_from_path(str(pdf_path), dpi=CONFIG['dpi'])
        except Exception as e:
            raise Exception(f"PDF conversion error: {e}")
        
        all_text_parts = []
        
        # Process each page
        for page_num, image in enumerate(images, 1):
            try:
                # Apply preprocessing if needed
                if 'preprocessed' in method:
                    image = self.preprocess_image(image)
                
                # Run OCR
                if 'tesseract' in method:
                    page_text = self.ocr_tesseract(image)
                elif 'deepseek' in method:
                    page_text = self.ocr_deepseek(image)
                else:
                    page_text = ""
                
                all_text_parts.append(page_text)
            
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
                all_text_parts.append(f"[ERROR ON PAGE {page_num}]")
        
        # Combine all pages
        full_text = "\n\n".join(all_text_parts)
        
        # Apply Llama cleanup if needed
        if 'llama' in method:
            full_text = self.cleanup_with_llama(full_text)
        
        return full_text
    
    # ============================================
    # ACCURACY CALCULATION
    # ============================================
    
    def calculate_accuracy(self, ground_truth: str, extracted: str) -> Dict[str, float]:
        """
        Calculate accuracy metrics between ground truth and extracted text
        
        Args:
            ground_truth: Manually extracted text
            extracted: OCR extracted text
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Normalize texts
        gt_normalized = ground_truth.lower().strip()
        ex_normalized = extracted.lower().strip()
        
        # Character-level accuracy
        char_similarity = difflib.SequenceMatcher(None, gt_normalized, ex_normalized).ratio()
        
        # Word-level accuracy
        gt_words = gt_normalized.split()
        ex_words = ex_normalized.split()
        word_similarity = difflib.SequenceMatcher(None, gt_words, ex_words).ratio()
        
        # Line-level accuracy
        gt_lines = [line.strip() for line in gt_normalized.split('\n') if line.strip()]
        ex_lines = [line.strip() for line in ex_normalized.split('\n') if line.strip()]
        line_similarity = difflib.SequenceMatcher(None, gt_lines, ex_lines).ratio()
        
        # Levenshtein distance
        levenshtein_distance = Levenshtein.distance(gt_normalized, ex_normalized)
        max_len = max(len(gt_normalized), len(ex_normalized))
        normalized_levenshtein = 1 - (levenshtein_distance / max_len) if max_len > 0 else 1.0
        
        return {
            "character_similarity": round(char_similarity * 100, 2),
            "word_similarity": round(word_similarity * 100, 2),
            "line_similarity": round(line_similarity * 100, 2),
            "levenshtein_similarity": round(normalized_levenshtein * 100, 2),
            "average_similarity": round((char_similarity + word_similarity + normalized_levenshtein) / 3 * 100, 2)
        }
    
    # ============================================
    # MAIN PROCESSING PIPELINE
    # ============================================
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files from original_pdfs directory"""
        pdf_files = sorted(list(self.original_pdfs_dir.glob('*.pdf')))
        return pdf_files
    
    def verify_files(self) -> Tuple[List[Path], List[str]]:
        """
        Verify that all PDFs have corresponding manual extraction files
        
        Returns:
            Tuple of (valid_pdf_files, missing_files)
        """
        pdf_files = self.get_pdf_files()
        valid_files = []
        missing_files = []
        
        for pdf_file in pdf_files:
            manual_file = self.manual_extraction_dir / f"{pdf_file.stem}.txt"
            if manual_file.exists():
                valid_files.append(pdf_file)
            else:
                missing_files.append(pdf_file.name)
        
        return valid_files, missing_files
    
    def process_method(self, method: str, pdf_files: List[Path]):
        """
        Process all PDF files with a specific method
        
        Args:
            method: Method name (e.g., 'tesseract_preprocessed')
            pdf_files: List of PDF file paths to process
        """
        print(f"\n{'='*60}")
        print(f"Processing Method: {method} ({self.methods.index(method) + 1}/{len(self.methods)})")
        print(f"{'='*60}")
        
        # Create method output directory
        method_dir = self.ocr_extraction_dir / method
        method_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize method results
        method_results = []
        start_time = time.time()
        
        # Process each PDF
        for pdf_file in tqdm(pdf_files, desc=f"{method}", unit="file"):
            # Check if already processed
            if self.is_processed(method, pdf_file.name):
                print(f"  ✓ Skipping {pdf_file.name} (already processed)")
                continue
            
            try:
                # Process PDF
                extracted_text = self.process_pdf_with_method(pdf_file, method)
                
                # Save extracted text
                output_file = method_dir / f"{pdf_file.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                
                # Load ground truth
                ground_truth_file = self.manual_extraction_dir / f"{pdf_file.stem}.txt"
                with open(ground_truth_file, 'r', encoding='utf-8') as f:
                    ground_truth = f.read()
                
                # Calculate accuracy
                accuracy = self.calculate_accuracy(ground_truth, extracted_text)
                
                # Store results
                method_results.append({
                    'filename': pdf_file.name,
                    'accuracy': accuracy
                })
                
                # Update progress
                self.update_progress(method, pdf_file.name, len(pdf_files))
                
                print(f"  ✓ {pdf_file.name}: {accuracy['average_similarity']:.2f}% accuracy")
            
            except Exception as e:
                print(f"  ✗ Error processing {pdf_file.name}: {e}")
                self.log_error(method, pdf_file.name, str(e))
                continue
        
        # Calculate method statistics
        processing_time = time.time() - start_time
        
        if method_results:
            accuracies = [r['accuracy']['average_similarity'] for r in method_results]
            method_stats = {
                'average_accuracy': round(sum(accuracies) / len(accuracies), 2),
                'min_accuracy': round(min(accuracies), 2),
                'max_accuracy': round(max(accuracies), 2),
                'processing_time_seconds': round(processing_time, 2),
                'files_processed': len(method_results)
            }
        else:
            method_stats = {
                'average_accuracy': 0,
                'min_accuracy': 0,
                'max_accuracy': 0,
                'processing_time_seconds': round(processing_time, 2),
                'files_processed': 0
            }
        
        # Save method results
        self.results_data['methods'][method] = method_stats
        
        # Add to detailed results
        for result in method_results:
            # Find or create entry for this file
            file_entry = next(
                (item for item in self.results_data['detailed_results'] if item['filename'] == result['filename']),
                None
            )
            
            if file_entry is None:
                file_entry = {'filename': result['filename'], 'methods': {}}
                self.results_data['detailed_results'].append(file_entry)
            
            file_entry['methods'][method] = result['accuracy']
        
        self.save_results()
        
        # Print summary
        print(f"\n✓ {method} complete")
        print(f"  Average Accuracy: {method_stats['average_accuracy']:.2f}%")
        print(f"  Processing Time: {processing_time/60:.1f} minutes")
        print(f"  Files: {method_stats['files_processed']}/{len(pdf_files)}")
    
    def run_pipeline(self):
        """Run the complete OCR comparison pipeline"""
        print("="*60)
        print("OCR COMPARISON PIPELINE")
        print("="*60)
        
        # Verify files
        print("\nVerifying files...")
        valid_files, missing_files = self.verify_files()
        
        print(f"  Found {len(valid_files)} valid PDF files")
        
        if missing_files:
            print(f"  Warning: {len(missing_files)} PDFs missing manual extraction:")
            for missing in missing_files[:5]:  # Show first 5
                print(f"    - {missing}")
            if len(missing_files) > 5:
                print(f"    ... and {len(missing_files) - 5} more")
        
        if not valid_files:
            print("  ERROR: No valid files to process!")
            return
        
        # Update metadata
        self.results_data['metadata']['total_files'] = len(valid_files)
        self.save_results()
        
        # Process each method
        for method in self.methods:
            self.process_method(method, valid_files)
        
        # Generate comparison CSV
        self.generate_comparison_csv()
        
        # Final summary
        self.print_final_summary()
    
    def generate_comparison_csv(self):
        """Generate CSV file with comparison across all methods"""
        csv_path = self.ocr_extraction_dir / 'comparison_report.csv'
        
        # Prepare data
        rows = []
        for result in self.results_data['detailed_results']:
            row = {'filename': result['filename']}
            for method in self.methods:
                if method in result.get('methods', {}):
                    row[method] = result['methods'][method]['average_similarity']
                else:
                    row[method] = None
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Comparison report saved to: {csv_path}")
    
    def print_final_summary(self):
        """Print final summary of all methods"""
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        # Sort methods by average accuracy
        sorted_methods = sorted(
            self.results_data['methods'].items(),
            key=lambda x: x[1]['average_accuracy'],
            reverse=True
        )
        
        print("\nMethod Rankings (by average accuracy):")
        print("-"*60)
        for rank, (method, stats) in enumerate(sorted_methods, 1):
            print(f"{rank}. {method:30s} {stats['average_accuracy']:6.2f}% avg")
        
        print("\n" + "="*60)
        print("Pipeline complete!")
        print(f"Results saved to: {self.results_file}")
        print(f"Comparison CSV: {self.ocr_extraction_dir / 'comparison_report.csv'}")
        print("="*60)


def main():
    """Main entry point"""
    comparator = OCRMethodComparator()
    comparator.run_pipeline()


if __name__ == "__main__":
    main()