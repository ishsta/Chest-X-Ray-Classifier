"""
Simple verification script for the data pipeline outputs.
"""

import pandas as pd
import pickle
from pathlib import Path

def verify_pipeline():
    print("=" * 60)
    print("Data Pipeline Verification")
    print("=" * 60)
    
    # Check filtered CSV
    csv_path = Path("data/filtered_data.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\n✅ Filtered CSV found: {csv_path}")
        print(f"   Total samples: {len(df):,}")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        
        # Check labels
        if 'Binary_Label' in df.columns:
            label_dist = df['Binary_Label'].value_counts()
            print(f"\n📊 Label distribution:")
            print(f"   Normal (0): {label_dist.get(0, 0):,}")
            print(f"   Abnormal (1): {label_dist.get(1, 0):,}")
    else:
        print(f"\n❌ Filtered CSV not found: {csv_path}")
    
    # Check splits file
    splits_path = Path("data/splits.pkl")
    if splits_path.exists():
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
        
        print(f"\n✅ Splits file found: {splits_path}")
        print(f"   Train samples: {len(splits['train_indices']):,}")
        print(f"   Val samples: {len(splits['val_indices']):,}")
        print(f"   Test samples: {len(splits['test_indices']):,}")
    else:
        print(f"\n❌ Splits file not found: {splits_path}")
    
    # Check images directory
    images_dir = Path("data/images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png"))
        print(f"\n✅ Images directory found: {images_dir}")
        print(f"   Total images: {len(image_files):,}")
    else:
        print(f"\n❌ Images directory not found: {images_dir}")
    
    print("\n" + "=" * 60)
    print("✨ Verification complete!")
    print("=" * 60)

if __name__ == "__main__":
    verify_pipeline()
