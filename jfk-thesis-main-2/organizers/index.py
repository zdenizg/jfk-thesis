import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import pandas as pd
from PyPDF2 import PdfReader
import time
from pathlib import Path

# Configuration
BASE_URL = "https://www.archives.gov/research/jfk/release-2025"
DOWNLOAD_FOLDER = "/Users/furkandemir/Desktop/Thesis/files"
EXCEL_FILE = "/Users/furkandemir/Desktop/Thesis/jfk_documents_index.xlsx"

# Create download folder if it doesn't exist
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def get_pdf_page_count(pdf_path):
    """Get the number of pages in a PDF file"""
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

def download_file(url, destination):
    """Download a file from URL to destination"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def scrape_jfk_documents():
    """Main scraping function"""
    print("Fetching the main page...")
    
    try:
        response = requests.get(BASE_URL, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching main page: {e}")
        return
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all PDF links
    pdf_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.endswith('.pdf'):
            full_url = urljoin(BASE_URL, href)
            pdf_links.append(full_url)
    
    print(f"Found {len(pdf_links)} PDF files to download")
    
    # Data for Excel
    documents_data = []
    
    # Download each PDF and collect information
    for idx, pdf_url in enumerate(pdf_links, 1):
        # Extract filename from URL
        filename = pdf_url.split('/')[-1]
        local_path = os.path.join(DOWNLOAD_FOLDER, filename)
        
        print(f"[{idx}/{len(pdf_links)}] Downloading: {filename}")
        
        # Skip if already downloaded
        if os.path.exists(local_path):
            print(f"  -> Already exists, skipping download")
        else:
            if not download_file(pdf_url, local_path):
                print(f"  -> Failed to download")
                continue
            # Small delay to be respectful to the server
            time.sleep(0.5)
        
        # Get page count
        print(f"  -> Reading PDF to count pages...")
        page_count = get_pdf_page_count(local_path)
        
        # Add to data list
        documents_data.append({
            'document_name': filename,
            'number_of_pages': page_count if page_count else 'Error',
            'download_link': pdf_url,
            'local_path': local_path
        })
        
        print(f"  -> Complete! ({page_count} pages)")
    
    # Create Excel file
    print(f"\nCreating Excel file: {EXCEL_FILE}")
    df = pd.DataFrame(documents_data)
    
    # Create Excel writer with formatting
    with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='JFK Documents')
        
        # Get the worksheet
        worksheet = writer.sheets['JFK Documents']
        
        # Adjust column widths
        worksheet.column_dimensions['A'].width = 50  # document_name
        worksheet.column_dimensions['B'].width = 15  # number_of_pages
        worksheet.column_dimensions['C'].width = 80  # download_link
        worksheet.column_dimensions['D'].width = 60  # local_path
    
    print(f"\nComplete!")
    print(f"Downloaded {len(documents_data)} files to: {DOWNLOAD_FOLDER}")
    print(f"Excel index created at: {EXCEL_FILE}")
    print(f"\nSummary:")
    print(f"  Total documents: {len(documents_data)}")
    total_pages = sum([d['number_of_pages'] for d in documents_data if isinstance(d['number_of_pages'], int)])
    print(f"  Total pages: {total_pages}")

if __name__ == "__main__":
    print("JFK Documents Scraper")
    print("=" * 50)
    scrape_jfk_documents()