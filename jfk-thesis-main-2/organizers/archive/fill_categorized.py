#!/usr/bin/env python3
"""
This script counts filename repetitions in jfk_categorization_combined.csv
and fills the 'categorized' column in jfk-document-index.csv with those counts.
"""

import pandas as pd
from collections import Counter
import os

def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_csv_path = os.path.join(base_dir, "jfk-document-index.csv")
    combined_csv_path = os.path.join(base_dir, "results", "jfk_categorization_combined.csv")
    
    print("Loading CSV files...")
    
    # Read the categorization combined CSV to count filename occurrences
    combined_df = pd.read_csv(combined_csv_path)
    
    # Count occurrences of each filename
    filename_counts = Counter(combined_df['filename'])
    print(f"Found {len(filename_counts)} unique filenames in categorization file")
    print(f"Total rows in categorization file: {len(combined_df)}")
    
    # Read the index CSV
    index_df = pd.read_csv(index_csv_path)
    print(f"Total documents in index file: {len(index_df)}")
    
    # Fill the 'categorized' column with the count for each filename
    index_df['categorized'] = index_df['filename'].apply(lambda x: filename_counts.get(x, 0))
    
    # Count how many documents have been categorized (count > 0)
    categorized_count = (index_df['categorized'] > 0).sum()
    uncategorized_count = (index_df['categorized'] == 0).sum()
    
    print(f"\nResults:")
    print(f"  - Documents with categorization: {categorized_count}")
    print(f"  - Documents without categorization: {uncategorized_count}")
    
    # Save the updated index CSV
    index_df.to_csv(index_csv_path, index=False)
    print(f"\nUpdated '{index_csv_path}' successfully!")
    
    # Show some examples
    print("\nSample of documents with categorization counts:")
    sample = index_df[index_df['categorized'] > 0].head(10)[['filename', 'number_of_pages', 'categorized']]
    print(sample.to_string(index=False))

if __name__ == "__main__":
    main()
