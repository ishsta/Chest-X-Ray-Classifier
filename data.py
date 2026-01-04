"""
Data Preparation and Dataset Module

Handles:
- Loading and filtering NIH Chest X-ray dataset
- Creating stratified train/val/test splits
- PyTorch Dataset implementation
- DataLoader creation with transforms
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter
from torchvision import transforms


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

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
    print(f"📂 Loading data from {csv_path}...")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    print(f"   Total entries in CSV: {initial_count:,}")
    
    # Get actual image files
    images_dir = Path(images_dir)
    available_images = set()
    
    # Check for images in the directory
    if images_dir.exists():
        for img_path in images_dir.glob('*.png'):
            available_images.add(img_path.name)
    
    print(f"   Available images found: {len(available_images):,}")
    
    # Filter dataframe
    df_filtered = df[df['Image Index'].isin(available_images)].copy()
    filtered_count = len(df_filtered)
    
    print(f"   Filtered entries: {filtered_count:,} ({filtered_count/initial_count*100:.1f}%)")
    
    # Create binary labels
    # "No Finding" = Normal (0), everything else = Abnormal (1)
    df_filtered['Binary_Label'] = (df_filtered['Finding Labels'] != 'No Finding').astype(int)
    
    # Display class distribution
    normal_count = (df_filtered['Binary_Label'] == 0).sum()
    abnormal_count = (df_filtered['Binary_Label'] == 1).sum()
    
    print(f"\n   Class Distribution:")
    print(f"      Normal (0):   {normal_count:,} ({normal_count/filtered_count*100:.1f}%)")
    print(f"      Abnormal (1): {abnormal_count:,} ({abnormal_count/filtered_count*100:.1f}%)")
    
    # Save filtered CSV
    if save_filtered:
        output_path = Path(csv_path).parent / 'filtered_data.csv'
        df_filtered.to_csv(output_path, index=False)
        print(f"\n   💾 Saved filtered data to: {output_path}")
    
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
    print(f"\n🔀 Creating stratified splits...")
    print(f"   Ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['Binary_Label'],
        random_state=random_state
    )
    
    # Second split: separate train and validation
    # Adjust validation ratio for remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df['Binary_Label'],
        random_state=random_state
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"\n   ✅ Split complete!")
    print(f"      Train: {len(train_df):,} samples")
    print(f"      Val:   {len(val_df):,} samples")
    print(f"      Test:  {len(test_df):,} samples")
    
    return train_df, val_df, test_df


def verify_data_split(train_df, val_df, test_df):
    """
    Verify and display statistics for data splits.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe
        test_df (pd.DataFrame): Test dataframe
    """
    print("\n" + "="*60)
    print("DATA SPLIT VERIFICATION")
    print("="*60)
    
    def get_distribution(df, name):
        """Helper to get class distribution."""
        total = len(df)
        normal = (df['Binary_Label'] == 0).sum()
        abnormal = (df['Binary_Label'] == 1).sum()
        
        print(f"\n{name}:")
        print(f"  Total: {total:,}")
        print(f"  Normal (0):   {normal:,} ({normal/total*100:.2f}%)")
        print(f"  Abnormal (1): {abnormal:,} ({abnormal/total*100:.2f}%)")
        
        return normal, abnormal
    
    # Get distributions
    train_stats = get_distribution(train_df, "Training Set")
    val_stats = get_distribution(val_df, "Validation Set")
    test_stats = get_distribution(test_df, "Test Set")
    
    # Overall statistics
    total_samples = len(train_df) + len(val_df) + len(test_df)
    total_normal = train_stats[0] + val_stats[0] + test_stats[0]
    total_abnormal = train_stats[1] + val_stats[1] + test_stats[1]
    
    print(f"\nOverall:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Normal:   {total_normal:,} ({total_normal/total_samples*100:.2f}%)")
    print(f"  Abnormal: {total_abnormal:,} ({total_abnormal/total_samples*100:.2f}%)")
    
    print("\n" + "="*60)


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
        'test_indices': test_df.index.tolist()
    }
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(splits, f)
    
    print(f"\n💾 Saved split indices to: {save_path}")


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH Chest X-Ray binary classification.
    
    Args:
        dataframe (pd.DataFrame): DataFrame with 'Image Index' and 'Binary_Label' columns
        images_dir (str or Path): Directory containing the chest X-ray images
        transform (callable, optional): Optional transform to apply to images
        image_size (int): Size to resize images to (only used if transform is None)
    """
    
    def __init__(self, dataframe, images_dir, transform=None, image_size=224):
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.image_size = image_size
        
        # If no transform provided, use basic transform
        if self.transform is None:
            self.transform = get_default_transform(image_size, is_train=False)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            tuple: (image_tensor, label)
                - image_tensor: torch.Tensor of shape (3, H, W)
                - label: int (0 for Normal, 1 for Abnormal)
        """
        # Get image filename and label
        row = self.dataframe.iloc[idx]
        img_name = row['Image Index']
        label = row['Binary_Label']
        
        # Load image
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Get the distribution of classes in this dataset."""
        labels = self.dataframe['Binary_Label'].values
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

def get_train_transform(image_size=224):
    """
    Get training transforms with data augmentation.
    
    Includes:
    - Resize to (image_size, image_size)
    - Random horizontal flip
    - Random rotation (±10 degrees)
    - Optional random affine
    - Convert to tensor
    - Normalize with ImageNet stats
    
    Args:
        image_size (int): Target image size
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transform(image_size=224):
    """
    Get validation/test transforms (deterministic, no augmentation).
    
    Includes:
    - Resize to (image_size, image_size)
    - Convert to tensor
    - Normalize with ImageNet stats
    
    Args:
        image_size (int): Target image size
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_default_transform(image_size=224, is_train=False):
    """
    Get default transform based on split type.
    
    Args:
        image_size (int): Target image size
        is_train (bool): Whether this is for training
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if is_train:
        return get_train_transform(image_size)
    else:
        return get_val_transform(image_size)


# ============================================================================
# DATALOADER CREATION
# ============================================================================

def create_dataloaders(train_df, val_df, test_df, images_dir, 
                       batch_size=32, num_workers=4, image_size=224):
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe
        test_df (pd.DataFrame): Test dataframe
        images_dir (str or Path): Directory containing images
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        image_size (int): Image size for transforms
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("\n📦 Creating DataLoaders...")
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        train_df,
        images_dir,
        transform=get_train_transform(image_size)
    )
    
    val_dataset = ChestXrayDataset(
        val_df,
        images_dir,
        transform=get_val_transform(image_size)
    )
    
    test_dataset = ChestXrayDataset(
        test_df,
        images_dir,
        transform=get_val_transform(image_size)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"   ✅ DataLoaders created!")
    print(f"      Train batches: {len(train_loader)}")
    print(f"      Val batches:   {len(val_loader)}")
    print(f"      Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# TESTING
# ============================================================================

def main():
    """Demo/test function."""
    print("Testing Data Module")
    print("=" * 60)
    
    # Test paths (adjust as needed)
    csv_path = 'data/Data_Entry_2017.csv'
    images_dir = 'data/images'
    
    if not Path(csv_path).exists():
        print(f"⚠️  CSV not found at {csv_path}")
        print("   This is a demo. Please adjust paths to test.")
        return
    
    # Load and filter data
    df = load_and_filter_data(csv_path, images_dir, save_filtered=True)
    
    # Create splits
    train_df, val_df, test_df = create_stratified_split(df)
    
    # Verify splits
    verify_data_split(train_df, val_df, test_df)
    
    # Save splits
    save_split_indices(train_df, val_df, test_df, 'data/splits.pkl')
    
    # Test dataset
    print("\n📋 Testing Dataset...")
    dataset = ChestXrayDataset(train_df, images_dir)
    print(f"   Dataset size: {len(dataset)}")
    
    # Get one sample
    image, label = dataset[0]
    print(f"   Sample image shape: {image.shape}")
    print(f"   Sample label: {label}")
    
    # Get class distribution
    dist = dataset.get_class_distribution()
    print(f"   Class distribution: {dist}")
    
    print("\n✨ Data module test complete!")


if __name__ == "__main__":
    main()
