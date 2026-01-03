"""
Training utilities for Chest X-Ray Classification

Includes:
- Device detection (GPU/CPU)
- Training loop for one epoch
- Evaluation function with metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def get_device(prefer_gpu=True):
    """
    Get the best available device.
    
    Args:
        prefer_gpu (bool): Whether to prefer GPU if available
        
    Returns:
        torch.device: Device to use for training
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    return device


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None):
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch (int, optional): Current epoch number for display
        
    Returns:
        dict: Training metrics (loss, accuracy)
    """
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Progress bar
    desc = f"Epoch {epoch}" if epoch is not None else "Training"
    pbar = tqdm(dataloader, desc=desc, leave=True)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels)
    }


def evaluate_model(model, dataloader, criterion, device, desc="Evaluating"):
    """
    Evaluate model on validation/test set.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
        desc (str): Description for progress bar
        
    Returns:
        dict: Evaluation metrics (loss, accuracy, precision, recall, f1)
    """
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=True)
        
        for images, labels in pbar:
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    eval_loss = running_loss / len(dataloader)
    eval_acc = accuracy_score(all_labels, all_preds)
    
    # Precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return {
        'loss': eval_loss,
        'accuracy': eval_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels)
    }


def create_optimizer(model, optimizer_type='adamw', lr=1e-4, weight_decay=0.01):
    """
    Create optimizer for model.
    
    Args:
        model (nn.Module): Model to optimize
        optimizer_type (str): Type of optimizer ('adam', 'adamw', 'sgd')
        lr (float): Learning rate
        weight_decay (float): Weight decay for regularization
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, scheduler_type='step', step_size=5, gamma=0.1):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type (str): Type of scheduler ('step', 'cosine', 'plateau')
        step_size (int): Step size for StepLR
        gamma (float): Multiplicative factor for learning rate decay
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=step_size
        )
    elif scheduler_type.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=gamma,
            patience=3
        )
    else:
        return None
    
    return scheduler


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer: Optimizer state
        epoch (int): Current epoch
        metrics (dict): Training metrics
        filepath (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath (str): Path to checkpoint
        model (nn.Module): Model to load weights into
        optimizer (optional): Optimizer to load state into
        
    Returns:
        dict: Checkpoint data
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def main():
    """Quick test of training utilities."""
    print("Testing training utilities...")
    
    # Test device detection
    device = get_device()
    
    # Create a simple model
    from models.vit import create_vit_small
    model = create_vit_small()
    model = model.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(model, 'adamw', lr=1e-4)
    print(f"\n✅ Optimizer created: {type(optimizer).__name__}")
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, 'step')
    print(f"✅ Scheduler created: {type(scheduler).__name__}")
    
    print("\n✨ Training utilities ready!")


if __name__ == "__main__":
    main()
