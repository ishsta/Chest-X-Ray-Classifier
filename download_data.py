"""
Kaggle Data Downloader for NIH Chest X-Ray Dataset

This script downloads the required data files from Kaggle:
- images_002.zip (~10,000 chest X-ray images)
- Data_Entry_2017.csv (annotations file)

Prerequisites:
1. Install kaggle package: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token" to download kaggle.json
   - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)
   - On Linux/Mac: chmod 600 ~/.kaggle/kaggle.json

Usage:
    python download_data.py
"""

import os
import zipfile
import subprocess
import sys
from pathlib import Path


def check_kaggle_setup():
    """Check if Kaggle API is properly configured."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("❌ Kaggle API not configured!")
        print("\n📋 Setup Instructions:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section and click 'Create New API Token'")
        print("3. This downloads 'kaggle.json'")
        print(f"4. Move kaggle.json to: {kaggle_dir}")
        
        if sys.platform != 'win32':
            print(f"5. Run: chmod 600 {kaggle_json}")
        
        return False
    
    print("✅ Kaggle API credentials found")
    return True


def download_from_kaggle(dataset, filename, output_dir):
    """Download a specific file from a Kaggle dataset."""
    print(f"\n📥 Downloading {filename}...")
    
    cmd = [
        'kaggle', 'datasets', 'download',
        '-d', dataset,
        '-f', filename,
        '-p', output_dir,
        '--force'  # Overwrite if exists
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Downloaded {filename}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading {filename}:")
        print(e.stderr)
        return False


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"\n📦 Extracting {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted to {extract_to}")
        
        # Remove zip file after extraction
        zip_path.unlink()
        print(f"🗑️  Removed {zip_path.name}")
        return True
    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return False


def main():
    """Main download script."""
    print("=" * 60)
    print("NIH Chest X-Ray Dataset Downloader")
    print("=" * 60)
    
    # Configuration
    DATASET = "nih-chest-xrays/data"
    DATA_DIR = Path("data")
    
    # Files to download
    FILES = [
        "images_002.zip",
        "Data_Entry_2017.csv"
    ]
    
    # Step 1: Check Kaggle setup
    if not check_kaggle_setup():
        print("\n⚠️  Please set up Kaggle API credentials first!")
        sys.exit(1)
    
    # Step 2: Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    print(f"\n📁 Data directory: {DATA_DIR.absolute()}")
    
    # Step 3: Download files
    for filename in FILES:
        file_path = DATA_DIR / filename
        
        # Skip if already exists (unless it's a zip with no extracted folder)
        if file_path.exists() and not filename.endswith('.zip'):
            print(f"\n⏭️  {filename} already exists, skipping...")
            continue
        
        # For zip files, check if extracted folder exists
        if filename.endswith('.zip'):
            extracted_folder = DATA_DIR / filename.replace('.zip', '')
            if extracted_folder.exists():
                print(f"\n⏭️  {extracted_folder.name} already extracted, skipping...")
                continue
        
        # Download file
        success = download_from_kaggle(DATASET, filename, str(DATA_DIR))
        
        if not success:
            print(f"\n⚠️  Failed to download {filename}")
            continue
        
        # Extract if it's a zip file
        if filename.endswith('.zip'):
            zip_path = DATA_DIR / filename
            if zip_path.exists():
                extract_zip(zip_path, DATA_DIR)
    
    # Step 4: Verify downloads
    print("\n" + "=" * 60)
    print("📊 Download Summary")
    print("=" * 60)
    
    csv_path = DATA_DIR / "Data_Entry_2017.csv"
    images_dir = DATA_DIR / "images_002"
    
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"✅ CSV file: {csv_path.name} ({size_mb:.2f} MB)")
    else:
        print(f"❌ CSV file not found: {csv_path}")
    
    if images_dir.exists():
        num_images = len(list(images_dir.glob("*.png")))
        print(f"✅ Images folder: {images_dir.name} ({num_images} images)")
    else:
        print(f"❌ Images folder not found: {images_dir}")
    
    print("\n✨ Data download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
