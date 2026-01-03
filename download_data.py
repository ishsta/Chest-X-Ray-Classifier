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


# Configuration
DATASET = "nih-chest-xrays/data"
DATA_DIR = Path("data")


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


def download_dataset_subset():
    """Download only the files we need from the full dataset."""
    print(f"\n📥 Downloading dataset subset...")
    
    # Download only images_002 and CSV
    cmd = [
        'kaggle', 'datasets', 'download',
        '-d', DATASET,
        '-p', str(DATA_DIR),
        '--unzip'
    ]
    
    try:
        print("⚠️  Note: This downloads the ENTIRE dataset (~45GB). This may take a while...")
        print("    For faster setup, you can manually download from:")
        print("    https://www.kaggle.com/datasets/nih-chest-xrays/data")
        
        user_input = input("\n⏸️  Continue with full download? (y/n): ")
        
        if user_input.lower() != 'y':
            print("\n❌ Download cancelled.")
            print("\n📝 Manual setup instructions:")
            print("1. Go to https://www.kaggle.com/datasets/nih-chest-xrays/data")
            print("2. Download only these files:")
            print("   - Data_Entry_2017.csv")
            print("   - images_002 folder (or download images_002.tar.gz and extract)")
            print(f"3. Place them in: {DATA_DIR.absolute()}")
            return False
            
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset")
        return False
    except KeyboardInterrupt:
        print("\n\n❌ Download cancelled by user")
        return False


def download_csv_only():
    """Download just the CSV file."""
    print(f"\n📥 Downloading CSV annotation file...")
    
    cmd = [
        'kaggle', 'datasets', 'download',
        '-d', DATASET,
        '-f', 'Data_Entry_2017.csv',
        '-p', str(DATA_DIR),
        '--unzip'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Downloaded Data_Entry_2017.csv")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False


def main():
    """Main download script."""
    print("=" * 60)
    print("NIH Chest X-Ray Dataset Downloader")
    print("=" * 60)
    
    # Step 1: Check Kaggle setup
    if not check_kaggle_setup():
        print("\n⚠️  Please set up Kaggle API credentials first!")
        sys.exit(1)
    
    # Step 2: Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    print(f"\n📁 Data directory: {DATA_DIR.absolute()}")
    
    # Step 3: Check what already exists
    csv_path = DATA_DIR / "Data_Entry_2017.csv"
    images_dir = DATA_DIR / "images_002" / "images"
    
    print("\n📊 Checking existing files...")
    csv_exists = csv_path.exists()
    images_exist = images_dir.exists() and len(list(images_dir.glob("*.png"))) > 0
    
    if csv_exists:
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"✅ CSV file exists: {csv_path.name} ({size_mb:.2f} MB)")
    
    if images_exist:
        num_images = len(list(images_dir.glob("*.png")))
        print(f"✅ Images exist: {images_dir} ({num_images} images)")
    
    # Step 4: Download what's missing
    if csv_exists and images_exist:
        print("\n✨ All required data already downloaded!")
        return
    
    print("\n" + "=" * 60)
    print("⚠️  IMPORTANT: Kaggle Dataset Size Information")
    print("=" * 60)
    print("The full NIH Chest X-ray dataset is ~45GB")
    print("We only need:")
    print("  - Data_Entry_2017.csv (~7.5MB)")
    print("  - images_002 folder (~4GB, ~10,000 images)")
    print("\nKaggle API doesn't support selective folder download.")
    print("=" * 60)
    
    # Download CSV if needed
    if not csv_exists:
        download_csv_only()
    
    # Check if images still needed
    if not images_exist:
        print("\n📁 For images_002, you have two options:")
        print("\n  Option 1: Manual download (RECOMMENDED)")
        print("    1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345")
        print("    2. Download 'images_002.tar.gz' (~4GB)")
        print(f"    3. Extract to: {DATA_DIR.absolute()}")
        print("\n  Option 2: Full Kaggle download (~45GB, not recommended)")
        print("    - Downloads entire dataset including all 12 image folders")
        
        choice = input("\n⏩ Skip image download for now? (y/n): ")
        
        if choice.lower() != 'y':
            download_dataset_subset()
    
    # Step 5: Final verification
    print("\n" + "=" * 60)
    print("📊 Download Summary")
    print("=" * 60)
    
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"✅ CSV file: {csv_path.name} ({size_mb:.2f} MB)")
    else:
        print(f"❌ CSV file not found: {csv_path}")
    
    if images_dir.exists():
        num_images = len(list(images_dir.glob("*.png")))
        print(f"✅ Images folder: {images_dir} ({num_images} images)")
    else:
        print(f"⚠️  Images folder not found: {images_dir}")
        print(f"    Please download manually from Box link above")
    
    print("\n✨ Setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

