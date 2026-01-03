"""
Custom PyTorch Dataset for NIH Chest X-Ray Classification

Implements ChestXrayDataset with:
- Image loading and preprocessing
- Configurable transforms for train vs val/test
- Binary classification labels (Normal vs Abnormal)
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
from torchvision import transforms


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH Chest X-Ray binary classification.
    
    Args:
        dataframe (pd.DataFrame): DataFrame with 'Image Index' and 'Binary_Label' columns
        images_dir (str or Path): Directory containing image files
        transform (callable, optional): Transform to apply to images
        image_size (int): Size to resize images to (default: 224)
    """
    
    def __init__(self, dataframe, images_dir, transform=None, image_size=224):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        
        # Check if images are in a subdirectory
        if (self.images_dir / "images").exists():
            self.images_dir = self.images_dir / "images"
        
        self.transform = transform
        self.image_size = image_size
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = get_default_transform(image_size, is_train=False)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            tuple: (image_tensor, label)
                - image_tensor: torch.Tensor of shape (3, H, W)
                - label: int (0 for Normal, 1 for Abnormal)
        """
        # Get image filename and label
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        label = int(row['Binary_Label'])
        
        # Load image
        img_path = self.images_dir / img_name
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Please ensure images_002 is downloaded and extracted properly."
            )
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Get the distribution of classes in this dataset."""
        counts = self.df['Binary_Label'].value_counts().to_dict()
        return {
            'normal': counts.get(0, 0),
            'abnormal': counts.get(1, 0),
            'total': len(self.df)
        }


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
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
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


def create_dataloaders(train_df, val_df, test_df, images_dir, 
                       batch_size=32, num_workers=4, image_size=224):
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe
        test_df (pd.DataFrame): Test dataframe
        images_dir (str or Path): Directory containing images
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of worker processes
        image_size (int): Image size for transforms
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    print("\n" + "=" * 60)
    print("Creating DataLoaders")
    print("=" * 60)
    
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
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Val:   {len(val_dataset):,} samples")
    print(f"   Test:  {len(test_dataset):,} samples")
    
    # Create dataloaders
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
    
    print(f"\n🔄 DataLoader configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Num workers: {num_workers}")
    print(f"   Image size: {image_size}x{image_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    print("\n✅ DataLoaders created successfully!")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader


def main():
    """Demo/test function."""
    import pandas as pd
    
    # Example usage - requires images to be downloaded
    print("Testing ChestXrayDataset...")
    
    # Try to load a sample
    csv_path = "data/filtered_data.csv"
    images_dir = "data/images_002"
    
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        
        # Take a small sample for testing
        sample_df = df.head(10)
        
        # Create dataset
        dataset = ChestXrayDataset(
            sample_df,
            images_dir,
            transform=get_train_transform()
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        try:
            # Try to load first item
            image, label = dataset[0]
            print(f"\n✅ Successfully loaded sample:")
            print(f"   Image shape: {image.shape}")
            print(f"   Label: {label} ({'Normal' if label == 0 else 'Abnormal'})")
            print(f"   Tensor dtype: {image.dtype}")
            print(f"   Tensor range: [{image.min():.3f}, {image.max():.3f}]")
            
            # Check class distribution
            dist = dataset.get_class_distribution()
            print(f"\n📊 Class distribution in sample:")
            print(f"   Normal: {dist['normal']}")
            print(f"   Abnormal: {dist['abnormal']}")
            
        except FileNotFoundError as e:
            print(f"\n⚠️  {e}")
            print("Please download images following DATA_DOWNLOAD.md")
            
    else:
        print(f"⚠️  Filtered CSV not found at {csv_path}")
        print("Run data_preparation.py first")


if __name__ == "__main__":
    main()
