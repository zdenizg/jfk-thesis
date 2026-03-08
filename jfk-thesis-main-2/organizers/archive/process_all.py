import subprocess
import time
import os
from pathlib import Path

# Define your folder structure
base_path = "/Users/furkandemir/Desktop/Thesis/files"
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Define folder groups based on your structure
folder_groups = {
    "single": ["1", "2", "3", "4", "5"],
    "grouped": ["6-10", "11-20", "21-30", "31-40", "41-50", 
                "51-60", "61-70", "71-80", "81-90", "91-100", "101+"]
}

def process_folder(folder_name):
    """Process a single folder."""
    print(f"\n{'='*60}")
    print(f"🚀 Processing folder: {folder_name}")
    print(f"{'='*60}\n")
    
    folder_path = f"{base_path}/{folder_name}"
    
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        return False
    
    result = subprocess.run(
        ["python", "categorizer.py", folder_name],
        capture_output=False
    )
    
    if result.returncode == 0:
        print(f"✅ Folder {folder_name} completed")
        # Move results to results folder
        for file in Path(".").glob(f"jfk_categorization_{folder_name}*"):
            file.rename(results_dir / file.name)
        return True
    else:
        print(f"❌ Folder {folder_name} failed")
        return False

def main():
    print("🎯 JFK Document Categorization - Batch Processor")
    print("="*60)
    
    completed = []
    failed = []
    
    # Process single folders first (likely smaller)
    print("\n📁 Processing single folders...")
    for folder in folder_groups["single"]:
        success = process_folder(folder)
        if success:
            completed.append(folder)
        else:
            failed.append(folder)
        time.sleep(5)  # Brief pause between folders
    
    # Process grouped folders
    print("\n📁 Processing grouped folders...")
    for folder in folder_groups["grouped"]:
        success = process_folder(folder)
        if success:
            completed.append(folder)
        else:
            failed.append(folder)
        time.sleep(5)
    
    # Summary
    print("\n" + "="*60)
    print("📊 PROCESSING SUMMARY")
    print("="*60)
    print(f"✅ Completed: {len(completed)} folders")
    print(f"❌ Failed: {len(failed)} folders")
    
    if failed:
        print(f"\nFailed folders: {', '.join(failed)}")
    
    print(f"\n📁 Results saved in: {results_dir.absolute()}")
    print("🎉 Batch processing complete!")

if __name__ == "__main__":
    main()