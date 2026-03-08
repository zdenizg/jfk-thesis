"""
PDF Text Scraper with OCR Support
Handles both text-based PDFs and scanned image PDFs
Exports results to Excel with document names
"""

import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
from pathlib import Path
import pandas as pd
from datetime import datetime


class PDFScraper:
    def __init__(self, tesseract_path=None):
        """
        Initialize PDF scraper
        
        Args:
            tesseract_path: Path to tesseract executable (optional)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def extract_text_pypdf(self, pdf_path):
        """
        Extract text using PyPDF2 (for text-based PDFs)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num} ---\n"
                        text += page_text
        except Exception as e:
            print(f"Error extracting text with PyPDF2: {e}")
        
        return text
    
    def extract_text_ocr(self, pdf_path, dpi=300, lang='eng'):
        """
        Extract text using OCR (for scanned PDFs)
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for converting PDF to images (default: 300)
            lang: Language for OCR (default: 'eng')
            
        Returns:
            Extracted text as string
        """
        text = ""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=dpi)
            
            # Process each page
            for page_num, image in enumerate(images, 1):
                print(f"Processing page {page_num}/{len(images)}...")
                
                # Perform OCR
                page_text = pytesseract.image_to_string(image, lang=lang)
                
                if page_text.strip():
                    text += f"\n--- Page {page_num} ---\n"
                    text += page_text
                    
        except Exception as e:
            print(f"Error extracting text with OCR: {e}")
        
        return text
    
    def extract_text_auto(self, pdf_path, ocr_threshold=100, **kwargs):
        """
        Automatically choose between PyPDF2 and OCR based on text content
        
        Args:
            pdf_path: Path to PDF file
            ocr_threshold: Minimum characters to consider PyPDF2 successful
            **kwargs: Additional arguments for OCR (dpi, lang)
            
        Returns:
            Extracted text as string
        """
        print(f"Processing: {pdf_path}")
        
        # Try PyPDF2 first (faster)
        text = self.extract_text_pypdf(pdf_path)
        
        # If insufficient text extracted, use OCR
        if len(text.strip()) < ocr_threshold:
            print("Insufficient text found with PyPDF2, using OCR...")
            text = self.extract_text_ocr(pdf_path, **kwargs)
        else:
            print("Text extracted successfully with PyPDF2")
        
        return text
    
    def scrape_pdf(self, pdf_path, output_path=None, method='auto', **kwargs):
        """
        Scrape text from PDF and optionally save to file
        
        Args:
            pdf_path: Path to PDF file
            output_path: Path to save extracted text (optional)
            method: Extraction method ('auto', 'pypdf', 'ocr')
            **kwargs: Additional arguments for OCR
            
        Returns:
            Extracted text as string
        """
        # Choose extraction method
        if method == 'auto':
            text = self.extract_text_auto(pdf_path, **kwargs)
        elif method == 'pypdf':
            text = self.extract_text_pypdf(pdf_path)
        elif method == 'ocr':
            text = self.extract_text_ocr(pdf_path, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Text saved to: {output_path}")
        
        return text
    
    def scrape_directory(self, directory_path, output_dir=None, **kwargs):
        """
        Scrape all PDFs in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            output_dir: Directory to save extracted text files
            **kwargs: Additional arguments for scraping
            
        Returns:
            Dictionary mapping PDF filenames to extracted text
        """
        directory = Path(directory_path)
        results = {}
        
        # Create output directory if needed
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process all PDF files
        pdf_files = list(directory.glob('*.pdf'))
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            print(f"\n{'='*60}")
            
            # Determine output path
            if output_dir:
                output_path = Path(output_dir) / f"{pdf_file.stem}.txt"
            else:
                output_path = None
            
            # Scrape PDF
            text = self.scrape_pdf(str(pdf_file), output_path, **kwargs)
            results[pdf_file.name] = text
        
        return results
    
    def scrape_to_excel(self, directory_path, excel_output_path, **kwargs):
        """
        Scrape all PDFs in directory and save to Excel file
        
        Args:
            directory_path: Path to directory containing PDFs
            excel_output_path: Path to save Excel file
            **kwargs: Additional arguments for scraping (method, dpi, lang)
        """
        directory = Path(directory_path)
        
        # Process all PDF files
        pdf_files = sorted(list(directory.glob('*.pdf')))
        print(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        if not pdf_files:
            print("No PDF files found!")
            return
        
        # Prepare data for Excel
        data = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                # Scrape PDF
                text = self.scrape_pdf(str(pdf_file), output_path=None, **kwargs)
                
                # Add to results
                data.append({
                    'Document_Name': pdf_file.name,
                    'Document_ID': pdf_file.stem,
                    'Extracted_Text': text.strip(),
                    'Character_Count': len(text.strip()),
                    'Processing_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                print(f"✓ Successfully extracted {len(text.strip())} characters")
                
            except Exception as e:
                print(f"✗ Error processing {pdf_file.name}: {e}")
                data.append({
                    'Document_Name': pdf_file.name,
                    'Document_ID': pdf_file.stem,
                    'Extracted_Text': f"ERROR: {str(e)}",
                    'Character_Count': 0,
                    'Processing_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame(data)
        df.to_excel(excel_output_path, index=False, engine='openpyxl')
        
        print(f"\n{'='*60}")
        print(f"✓ Results saved to: {excel_output_path}")
        print(f"Total documents processed: {len(data)}")
        print(f"Successful extractions: {sum(1 for d in data if d['Character_Count'] > 0)}")
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize scraper
    scraper = PDFScraper()
    
    # For Windows, you may need to specify tesseract path:
    # scraper = PDFScraper(tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    
    # For macOS (if installed via Homebrew):
    scraper = PDFScraper(tesseract_path='/opt/homebrew/bin/tesseract')
    
    # Process all PDFs and save to Excel
    input_directory = "/Users/furkandemir/Desktop/Thesis/files/1/simple"
    output_excel = "/Users/furkandemir/Desktop/Thesis/files/1/simple/extracted_texts.csv"
    
    # Run the scraper
    df = scraper.scrape_to_excel(
        directory_path=input_directory,
        excel_output_path=output_excel,
        method='auto',  # Use 'ocr' for scanned documents, 'auto' to detect automatically
        dpi=300,  # Higher DPI = better quality but slower (200-400 recommended)
        lang='eng'  # Language for OCR
    )
    
    # Display summary
    if df is not None:
        print("\n--- Summary ---")
        print(df[['Document_Name', 'Character_Count']].to_string())
    
    # Alternative: If you want to scrape a single PDF
    # single_pdf = "/Users/furkandemir/Desktop/Thesis/files/1/document.pdf"
    # if os.path.exists(single_pdf):
    #     text = scraper.scrape_pdf(single_pdf, method='ocr', dpi=300)
    #     print(text[:500])