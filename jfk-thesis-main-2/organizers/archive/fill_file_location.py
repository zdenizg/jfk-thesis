#!/usr/bin/env python3
"""
This script finds the file paths for each PDF in the index and fills 
the 'file_location' column in jfk-document-index.csv.
"""

import pandas as pd
import os
from pathlib import Path

def build_file_index(files_dir):
    """
    Build a dictionary mapping filenames to their full paths.
    This avoids repeatedly searching the directory structure.
    """
    file_index = {}
    
    print(f"Scanning directory: {files_dir}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(files_dir):
        for filename in files:
            if filename.endswith('.pdf'):
                full_path = os.path.join(root, filename)
                file_index[filename] = full_path
    
    print(f"Found {len(file_index)} PDF files")
    return file_index


def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    thesis_dir = os.path.dirname(base_dir)  # Parent directory (Thesis)
    
    index_csv_path = os.path.join(base_dir, "jfk-document-index.csv")
    files_dir = os.path.join(thesis_dir, "files")
    
    print("Building file index...")
    file_index = build_file_index(files_dir)
    
    print("\nLoading index CSV...")
    index_df = pd.read_csv(index_csv_path)
    print(f"Total documents in index: {len(index_df)}")
    
    # Fill the 'file_location' column
    def get_file_location(filename):
        return file_index.get(filename, "")
    
    index_df['file_location'] = index_df['filename'].apply(get_file_location)
    
    # Count results
    found_count = (index_df['file_location'] != "").sum()
    not_found_count = (index_df['file_location'] == "").sum()
    
    print(f"\nResults:")
    print(f"  - Files found: {found_count}")
    print(f"  - Files not found: {not_found_count}")
    
    # Save the updated index CSV
    index_df.to_csv(index_csv_path, index=False)
    print(f"\nUpdated '{index_csv_path}' successfully!")
    
    # Show some examples
    print("\nSample of documents with file locations:")
    sample = index_df[index_df['file_location'] != ""].head(10)[['filename', 'file_location']]
    for _, row in sample.iterrows():
        print(f"  {row['filename']}")
        print(f"    -> {row['file_location']}")
    
    # Show files not found (if any)
    if not_found_count > 0:
        print(f"\nFiles not found ({not_found_count}):")
        not_found = index_df[index_df['file_location'] == ""][['filename']].head(20)
        for _, row in not_found.iterrows():
            print(f"  - {row['filename']}")
        if not_found_count > 20:
            print(f"  ... and {not_found_count - 20} more")


if __name__ == "__main__":
    main()
