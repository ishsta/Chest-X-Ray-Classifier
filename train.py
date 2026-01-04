"""
Complete Training Pipeline for Chest X-Ray Classification

Includes:
- Device detection and training utilities
- Training and evaluation functions
- Federated learning (FedAvg algorithm)
- Evaluation metrics and visualization
- Main execution with CLI
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from pathlib import Path
import pickle
from copy import deepcopy
import argparse
import yaml

# Import from our consolidated modules
from model import create_vit_small, create_vit_base, create_lstm_small, create_lstm_base
from data import ChestXrayDataset, create_dataloaders, get_train_transform, get_val_transform


# ============================================================================
# DEVICE AND BASIC UTILITIES
# ============================================================================

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


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

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


# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_on_test_set(model, test_loader, device, save_dir='results'):
    """
    Comprehensive evaluation on test set.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device: Device to evaluate on
        save_dir (str): Directory to save results
        
    Returns:
        dict: Complete evaluation results
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n🧪 Evaluating on test set...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def plot_confusion_matrix(labels, predictions, save_path='results/confusion_matrix.png', 
                         class_names=['Normal', 'Abnormal']):
    """
    Create and save confusion matrix visualization.
    
    Args:
        labels (array): True labels
        predictions (array): Predicted labels
        save_path (str): Path to save plot
        class_names (list): Names of classes
    """
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved confusion matrix to: {save_path}")


def generate_classification_report(labels, predictions, 
                                   save_path='results/classification_report.txt',
                                   class_names=['Normal', 'Abnormal']):
    """
    Generate and save detailed classification report.
    
    Args:
        labels (array): True labels
        predictions (array): Predicted labels
        save_path (str): Path to save report
        class_names (list): Names of classes
    """
    report = classification_report(labels, predictions, target_names=class_names)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"   💾 Saved classification report to: {save_path}")


def visualize_predictions(model, test_loader, device, num_samples=20,
                         save_path='results/predictions.png',
                         class_names=['Normal', 'Abnormal']):
    """
    Visualize sample predictions with images.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device: Device to evaluate on
        num_samples (int): Number of samples to visualize
        save_path (str): Path to save visualization
        class_names (list): Names of classes
    """
    model.eval()
    
    images_list = []
    labels_list = []
    preds_list = []
    probs_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            images_list.extend(images.cpu())
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
            probs_list.extend(probs.cpu().numpy())
            
            if len(images_list) >= num_samples:
                break
    
    # Plot
    n_cols = 5
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx in range(min(num_samples, len(images_list))):
        img = images_list[idx].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        true_label = class_names[labels_list[idx]]
        pred_label = class_names[preds_list[idx]]
        confidence = probs_list[idx][preds_list[idx]]
        
        color = 'green' if labels_list[idx] == preds_list[idx] else 'red'
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                           color=color, fontsize=9)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved predictions visualization to: {save_path}")


def analyze_misclassified(model, test_loader, device,
                          save_path='results/misclassified.png',
                          class_names=['Normal', 'Abnormal'],
                          max_samples=16):
    """
    Find and visualize misclassified examples.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device: Device to evaluate on
        save_path (str): Path to save visualization
        class_names (list): Names of classes
        max_samples (int): Maximum number of misclassified samples to show
    """
    model.eval()
    
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    misclassified_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Find misclassified
            mask = (preds != labels.to(device))
            if mask.any():
                misclassified_images.extend(images[mask.cpu()].cpu())
                misclassified_labels.extend(labels[mask.cpu()].cpu().numpy())
                misclassified_preds.extend(preds[mask].cpu().numpy())
                misclassified_probs.extend(probs[mask].cpu().numpy())
            
            if len(misclassified_images) >= max_samples:
                break
    
    if len(misclassified_images) == 0:
        print("   🎉 No misclassifications found!")
        return
    
    # Plot
    num_to_show = min(max_samples, len(misclassified_images))
    n_cols = 4
    n_rows = (num_to_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx in range(num_to_show):
        img = misclassified_images[idx].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        true_label = class_names[misclassified_labels[idx]]
        pred_label = class_names[misclassified_preds[idx]]
        confidence = misclassified_probs[idx][misclassified_preds[idx]]
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                           color='red', fontsize=9)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_to_show, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Misclassified Examples', fontsize=14, y=1.00)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved misclassified examples to: {save_path}")
    print(f"   Found {len(misclassified_images)} misclassified samples")


def plot_training_history(history, save_path='results/training_history.png'):
    """
    Plot training history (for federated learning).
    
    Args:
        history (dict): Training history with rounds, losses, accuracies
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['rounds'], history['train_losses'], label='Train Loss', marker='o')
    if 'val_losses' in history:
        axes[0].plot(history['rounds'], history['val_losses'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Rounds')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'val_accuracies' in history:
        axes[1].plot(history['rounds'], history['val_accuracies'], label='Val Accuracy', 
                    marker='s', color='green')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy Over Rounds')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved training history to: {save_path}")


# ============================================================================
# FEDERATED LEARNING (FedAvg)
# ============================================================================

def partition_data_for_clients(dataset_indices, num_clients=3, allocation='balanced', seed=42):
    """
    Partition data indices among clients.
    
    Args:
        dataset_indices (list): List of dataset indices to partition
        num_clients (int): Number of clients
        allocation (str): 'balanced', 'random', or 'iid'
        seed (int): Random seed
        
    Returns:
        dict: Client ID -> list of indices
    """
    np.random.seed(seed)
    indices = np.array(dataset_indices)
    np.random.shuffle(indices)
    
    client_data = {}
    
    if allocation == 'balanced':
        # Evenly distribute data
        split_indices = np.array_split(indices, num_clients)
        for i in range(num_clients):
            client_data[f'client_{i}'] = split_indices[i].tolist()
    
    elif allocation in ['random', 'iid']:
        # Assign each sample randomly to a client
        client_assignments = np.random.randint(0, num_clients, size=len(indices))
        for i in range(num_clients):
            client_indices = indices[client_assignments == i]
            client_data[f'client_{i}'] = client_indices.tolist()
    
    else:
        raise ValueError(f"Unknown allocation strategy: {allocation}")
    
    return client_data


def create_client_dataloaders(full_dataset, client_data_map, batch_size=32, num_workers=4):
    """
    Create DataLoaders for each client.
    
    Args:
        full_dataset: Complete dataset
        client_data_map (dict): Client ID -> indices mapping
        batch_size (int): Batch size
        num_workers (int): Number of workers
        
    Returns:
        dict: Client ID -> DataLoader
    """
    client_loaders = {}
    
    for client_id, indices in client_data_map.items():
        subset = Subset(full_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        client_loaders[client_id] = loader
    
    return client_loaders


def get_model_weights(model):
    """
    Extract model weights as a dictionary.
    
    Args:
        model (nn.Module): Model to extract weights from
        
    Returns:
        dict: State dictionary of model weights
    """
    return deepcopy(model.state_dict())


def set_model_weights(model, weights):
    """
    Set model weights from a dictionary.
    
    Args:
        model (nn.Module): Model to update
        weights (dict): State dictionary of weights
    """
    model.load_state_dict(deepcopy(weights))


def federated_averaging(client_weights, client_sizes):
    """
    Aggregate client model weights using Federated Averaging.
    
    Args:
        client_weights (list): List of client state_dicts
        client_sizes (list): List of dataset sizes for each client
        
    Returns:
        dict: Averaged state dictionary
    """
    # Calculate total samples
    total_samples = sum(client_sizes)
    
    # Initialize averaged weights
    averaged_weights = {}
    
    # Get keys from first client
    keys = client_weights[0].keys()
    
    for key in keys:
        # Weighted average
        averaged_weights[key] = sum(
            client_weights[i][key] * (client_sizes[i] / total_samples)
            for i in range(len(client_weights))
        )
    
    return averaged_weights


def train_client(client_id, model, dataloader, criterion, optimizer, device, local_epochs=1):
    """
    Train a single client for local epochs.
    
    Args:
        client_id (str): Client identifier
        model (nn.Module): Model to train
        dataloader (DataLoader): Client's data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        local_epochs (int): Number of local epochs
        
    Returns:
        dict: Training metrics
    """
    model.train()
    total_loss = 0.0
    
    for epoch in range(local_epochs):
        epoch_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        total_loss += epoch_loss / len(dataloader)
    
    avg_loss = total_loss / local_epochs
    
    return {
        'client_id': client_id,
        'loss': avg_loss,
        'num_batches': len(dataloader)
    }


def train_federated(global_model, client_loaders, val_loader, num_rounds=5, 
                   local_epochs=2, lr=1e-4, device='cuda', save_dir='checkpoints'):
    """
    Execute federated learning training with FedAvg.
    
    Args:
        global_model (nn.Module): Initial global model
        client_loaders (dict): Client ID -> DataLoader mapping
        val_loader (DataLoader): Validation data loader
        num_rounds (int): Number of communication rounds
        local_epochs (int): Number of local training epochs per round
        lr (float): Learning rate
        device: Device to train on
        save_dir (str): Directory to save checkpoints
        
    Returns:
        dict: Training history
    """
    print(f"\n{'='*60}")
    print("FEDERATED TRAINING WITH FedAvg")
    print(f"{'='*60}")
    print(f"Clients: {len(client_loaders)}")
    print(f"Rounds: {num_rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Learning rate: {lr}")
    
    global_model = global_model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'rounds': [],
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': []
    }
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'─'*60}")
        print(f"📡 Round {round_num}/{num_rounds}")
        print(f"{'─'*60}")
        
        # Store client weights and sizes
        client_weights_list = []
        client_sizes = []
        round_losses = []
        
        # Train each client
        for client_id, client_loader in client_loaders.items():
            # Create local model copy
            local_model = deepcopy(global_model)
            local_model.to(device)
            
            # Create optimizer for local training
            optimizer = create_optimizer(local_model, 'adamw', lr=lr)
            
            # Train locally
            client_metrics = train_client(
                client_id, local_model, client_loader,
                criterion, optimizer, device, local_epochs
            )
            
            print(f"   {client_id}: Loss = {client_metrics['loss']:.4f} "
                  f"(Batches: {client_metrics['num_batches']})")
            
            # Store results
            client_weights_list.append(get_model_weights(local_model))
            client_sizes.append(len(client_loader.dataset))
            round_losses.append(client_metrics['loss'])
        
        # Aggregate weights using FedAvg
        print(f"\n   🔄 Aggregating client models...")
        global_weights = federated_averaging(client_weights_list, client_sizes)
        set_model_weights(global_model, global_weights)
        
        # Evaluate on validation set
        print(f"   📊 Evaluating global model...")
        val_metrics = evaluate_model(global_model, val_loader, criterion, device, desc="   Val")
        
        # Store history
        history['rounds'].append(round_num)
        history['train_losses'].append(np.mean(round_losses))
        history['val_losses'].append(val_metrics['loss'])
        history['val_accuracies'].append(val_metrics['accuracy'])
        
        print(f"\n   Round {round_num} Results:")
        print(f"      Avg Train Loss: {np.mean(round_losses):.4f}")
        print(f"      Val Loss:       {val_metrics['loss']:.4f}")
        print(f"      Val Accuracy:   {val_metrics['accuracy']:.4f}")
        print(f"      Val F1:         {val_metrics['f1']:.4f}")
        
        # Save checkpoint
        checkpoint_path = save_dir / f'federated_round_{round_num}.pth'
        save_checkpoint(global_model, None, round_num, val_metrics, checkpoint_path)
        print(f"      💾 Saved checkpoint to: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print("✨ Federated training complete!")
    print(f"{'='*60}")
    
    return history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_splits(splits_path='data/splits.pkl'):
    """Load pre-computed train/val/test splits."""
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    return splits


def create_model(model_type='vit', num_classes=2, size='small'):
    """
    Create model based on configuration.
    
    Args:
        model_type (str): 'vit' or 'lstm'
        num_classes (int): Number of output classes
        size (str): 'small' or 'base'
        
    Returns:
        nn.Module: Created model
    """
    if model_type == 'vit':
        if size == 'small':
            model = create_vit_small(num_classes=num_classes)
        else:
            model = create_vit_base(num_classes=num_classes)
    elif model_type == 'lstm':
        if size == 'small':
            model = create_lstm_small(num_classes=num_classes)
        else:
            model = create_lstm_base(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def run_federated_training(config, args):
    """Execute federated learning training."""
    print("\n" + "="*60)
    print("FEDERATED LEARNING MODE")
    print("="*60)
    
    # Load data
    print("\n📁 Loading data...")
    csv_path = config['data']['filtered_csv']
    images_dir = config['data']['images_dir']
    splits_path = config['data']['split_file']
    
    df = pd.read_csv(csv_path)
    splits = load_splits(splits_path)
    
    # Get split dataframes
    train_df = df.iloc[splits['train_indices']].reset_index(drop=True)
    val_df = df.iloc[splits['val_indices']].reset_index(drop=True)
    test_df = df.iloc[splits['test_indices']].reset_index(drop=True)
    
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val:   {len(val_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")
    
    # Create dataset
    train_dataset = ChestXrayDataset(
        train_df,
        images_dir,
        transform=get_train_transform(config['model']['image_size'])
    )
    
    # Partition data for clients
    num_clients = args.num_clients or config['federated']['num_clients']
    print(f"\n🌐 Partitioning data for {num_clients} clients...")
    
    client_data = partition_data_for_clients(
        list(range(len(train_dataset))),
        num_clients=num_clients,
        allocation=config['federated']['allocation']
    )
    
    for client_id, indices in client_data.items():
        print(f"   {client_id}: {len(indices):,} samples")
    
    # Create client dataloaders
    client_loaders = create_client_dataloaders(
        train_dataset,
        client_data,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # Create validation loader
    val_dataset = ChestXrayDataset(
        val_df,
        images_dir,
        transform=get_val_transform(config['model']['image_size'])
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Create model
    print(f"\n🤖 Creating {args.model or config['model']['type']} model...")
    model = create_model(
        model_type=args.model or config['model']['type'],
        num_classes=config['model']['num_classes'],
        size='small'  # Using small for faster training
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # Get device
    device = get_device()
    
    # Train with federated learning
    num_rounds = args.rounds or config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    lr = config['training']['learning_rate']
    
    history = train_federated(
        global_model=model,
        client_loaders=client_loaders,
        val_loader=val_loader,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        lr=lr,
        device=device,
        save_dir=config['deployment']['checkpoint_dir']
    )
    
    # Plot training history
    plot_training_history(history, save_path='results/federated_training_history.png')
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")
    
    test_dataset = ChestXrayDataset(
        test_df,
        images_dir,
        transform=get_val_transform(config['model']['image_size'])
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    model = model.to(device)
    results = evaluate_on_test_set(model, test_loader, device)
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    plot_confusion_matrix(results['labels'], results['predictions'])
    generate_classification_report(results['labels'], results['predictions'])
    visualize_predictions(model, test_loader, device, num_samples=20)
    analyze_misclassified(model, test_loader, device)
    
    # Save final model
    final_model_path = Path(config['deployment']['model_save_path']) / 'federated_final_model.pth'
    final_model_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"\n💾 Saved final model to: {final_model_path}")
    
    print(f"\n{'='*60}")
    print("✨ Federated training complete!")
    print(f"{'='*60}")


def run_centralized_training(config, args):
    """Execute centralized (non-federated) training."""
    print("\n" + "="*60)
    print("CENTRALIZED TRAINING MODE")
    print("="*60)
    print("\n⚠️  Note: Centralized training not yet implemented.")
    print("Use federated mode with --federated flag")


def main():
    parser = argparse.ArgumentParser(description='Chest X-Ray Classification')
    
    # Mode
    parser.add_argument('--federated', action='store_true',
                       help='Use federated learning')
    
    # Model
    parser.add_argument('--model', type=str, choices=['vit', 'lstm'],
                       help='Model architecture to use')
    
    # Federated settings
    parser.add_argument('--num_clients', type=int,
                       help='Number of federated clients')
    parser.add_argument('--rounds', type=int,
                       help='Number of communication rounds')
    
    # Config
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    # Evaluation only
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate a trained model')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    
    # Run training or evaluation
    if args.eval_only:
        print("Evaluation mode not yet implemented")
        return
    
    if args.federated:
        run_federated_training(config, args)
    else:
        run_centralized_training(config, args)


if __name__ == "__main__":
    main()
