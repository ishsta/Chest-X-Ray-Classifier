"""
Data Preparation Module

Handles loading, filtering, and splitting the NIH Chest X-ray dataset.

Functions:
- load_and_filter_data(): Load CSV and filter for available images
- create_stratified_split(): Create train/val/test splits
- verify_data_split(): Verify split statistics
- save_split_indices(): Save splits for reproducibility
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter


def load_and_filter_data(csv_path, images_dir, save_filtered=True):
    """
    Load the annotation CSV and filter for images that exist in images_dir.
    
    Args:
        csv_path (str or Path): Path to Data_Entry_2017.csv
        images_dir (str or Path): Directory containing images_002/images/
        save_filtered (bool): Whether to save filtered CSV
        
    Returns:
        pd.DataFrame: Filtered dataframe with only available images
    """
    print("=" * 60)
    print("Loading and Filtering Data")
    print("=" * 60)
    
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    
    # Load CSV
    print(f"\n📄 Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Total entries in CSV: {len(df):,}")
    
    # Get list of available images
    # Images are in images_002/images/*.png
    images_folder = images_dir / "images" if (images_dir / "images").exists() else images_dir
    
    print(f"\n📁 Scanning for images in: {images_folder}")
    available_images = set([f.name for f in images_folder.glob("*.png")])
    print(f"   Found {len(available_images):,} images")
    
    if len(available_images) == 0:
        print("\n⚠️  WARNING: No images found!")
        print("   Please download images_002 following DATA_DOWNLOAD.md instructions")
        print("   Returning full CSV anyway for preview...")
        return df
    
    # Filter dataframe
    print(f"\n🔍 Filtering dataframe for available images...")
    df_filtered = df[df['Image Index'].isin(available_images)].copy()
    print(f"   Entries after filtering: {len(df_filtered):,}")
    print(f"   Filtered out: {len(df) - len(df_filtered):,} entries")
    
    # Create binary label: Normal (0) vs Abnormal (1)
    print(f"\n🏷️  Creating binary labels...")
    df_filtered['Binary_Label'] = (df_filtered['Finding Labels'] != 'No Finding').astype(int)
    
    label_counts = df_filtered['Binary_Label'].value_counts()
    print(f"   Normal (0): {label_counts.get(0, 0):,} ({(label_counts.get(0, 0)/len(df_filtered)*100):.1f}%)")
    print(f"   Abnormal (1): {label_counts.get(1, 0):,} ({(label_counts.get(1, 0)/len(df_filtered)*100):.1f}%)")
    
    # Save filtered CSV
    if save_filtered and len(df_filtered) > 0:
        output_path = csv_path.parent / "filtered_data.csv"
        df_filtered.to_csv(output_path, index=False)
        print(f"\n💾 Saved filtered data to: {output_path}")
    
    print("\n✅ Data loading complete!")
    print("=" * 60)
    
    return df_filtered


def create_stratified_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Create stratified train/val/test splits.
    
    Args:
        df (pd.DataFrame): Dataframe with 'Binary_Label' column
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        test_ratio (float): Proportion for test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("\n" + "=" * 60)
    print("Creating Stratified Splits")
    print("=" * 60)
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    if 'Binary_Label' not in df.columns:
        raise ValueError("DataFrame must have 'Binary_Label' column")
    
    print(f"\n📊 Split ratios:")
    print(f"   Train: {train_ratio*100:.0f}%")
    print(f"   Val:   {val_ratio*100:.0f}%")
    print(f"   Test:  {test_ratio*100:.0f}%")
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['Binary_Label'],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df['Binary_Label'],
        random_state=random_state
    )
    
    print(f"\n✅ Split complete:")
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val:   {len(val_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")
    print(f"   Total: {len(train_df) + len(val_df) + len(test_df):,}")
    
    print("=" * 60)
    
    return train_df, val_df, test_df


def verify_data_split(train_df, val_df, test_df):
    """
    Verify and display statistics for data splits.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe
        test_df (pd.DataFrame): Test dataframe
    """
    print("\n" + "=" * 60)
    print("Split Verification")
    print("=" * 60)
    
    def get_distribution(df, name):
        """Helper to get class distribution."""
        counts = df['Binary_Label'].value_counts()
        total = len(df)
        return {
            'name': name,
            'total': total,
            'normal': counts.get(0, 0),
            'abnormal': counts.get(1, 0),
            'normal_pct': (counts.get(0, 0) / total * 100) if total > 0 else 0,
            'abnormal_pct': (counts.get(1, 0) / total * 100) if total > 0 else 0
        }
    
    splits = [
        get_distribution(train_df, 'Train'),
        get_distribution(val_df, 'Val'),
        get_distribution(test_df, 'Test')
    ]
    
    print(f"\n{'Split':<8} {'Total':>8} {'Normal':>8} {'(%)':>7} {'Abnormal':>10} {'(%)':>7}")
    print("-" * 60)
    
    for s in splits:
        print(f"{s['name']:<8} {s['total']:>8,} {s['normal']:>8,} {s['normal_pct']:>6.1f}% {s['abnormal']:>10,} {s['abnormal_pct']:>6.1f}%")
    
    # Check for data leakage
    train_images = set(train_df['Image Index'])
    val_images = set(val_df['Image Index'])
    test_images = set(test_df['Image Index'])
    
    overlap_train_val = train_images & val_images
    overlap_train_test = train_images & test_images
    overlap_val_test = val_images & test_images
    
    print(f"\n🔒 Data Leakage Check:")
    if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
        print("   ✅ No data leakage detected")
    else:
        print(f"   ⚠️  WARNING: Data leakage detected!")
        if overlap_train_val:
            print(f"      Train-Val overlap: {len(overlap_train_val)}")
        if overlap_train_test:
            print(f"      Train-Test overlap: {len(overlap_train_test)}")
        if overlap_val_test:
            print(f"      Val-Test overlap: {len(overlap_val_test)}")
    
    print("=" * 60)


def save_split_indices(train_df, val_df, test_df, save_path):
    """
    Save train/val/test split indices for reproducibility.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe
        test_df (pd.DataFrame): Test dataframe
        save_path (str or Path): Path to save pickle file
    """
    splits = {
        'train_indices': train_df.index.tolist(),
        'val_indices': val_df.index.tolist(),
        'test_indices': test_df.index.tolist(),
        'train_images': train_df['Image Index'].tolist(),
        'val_images': val_df['Image Index'].tolist(),
        'test_images': test_df['Image Index'].tolist()
    }
    
    save_path = Path(save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(splits, f)
    
    print(f"\n💾 Saved split indices to: {save_path}")


def main():
    """Demo/test function."""
    # Example usage
    csv_path = "data/Data_Entry_2017.csv"
    images_dir = "data/images"
    
    # Load and filter
    df = load_and_filter_data(csv_path, images_dir)
    
    if len(df) > 0:
        # Create splits
        train_df, val_df, test_df = create_stratified_split(df)
        
        # Verify
        verify_data_split(train_df, val_df, test_df)
        
        # Save
        save_split_indices(train_df, val_df, test_df, "data/splits.pkl")
        
        print("\n✨ Data preparation complete!")
    else:
        print("\n⚠️  Cannot create splits without images. Please download images first.")


if __name__ == "__main__":
    main()
