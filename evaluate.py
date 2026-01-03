"""
Evaluation and Visualization Tools

Includes:
- Model evaluation on test set
- Confusion matrix generation
- Classification metrics computation
- Prediction visualization
- Misclassified examples analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from pathlib import Path
import pandas as pd
from tqdm import tqdm


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
    from train import evaluate_model
    import torch.nn as nn
    
    print("=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    results = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        desc="Testing"
    )
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1']:.4f}")
    print(f"   Loss:      {results['loss']:.4f}")
    
    # Save metrics to file
    metrics_file = save_dir / "test_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("Test Set Evaluation Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall:    {results['recall']:.4f}\n")
        f.write(f"F1-Score:  {results['f1']:.4f}\n")
        f.write(f"Loss:      {results['loss']:.4f}\n")
    
    print(f"\n💾 Saved metrics to: {metrics_file}")
    
    return results


def plot_confusion_matrix(labels, predictions, save_path='results/confusion_matrix.png', class_names=['Normal', 'Abnormal']):
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
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved confusion matrix to: {save_path}")


def generate_classification_report(labels, predictions, save_path='results/classification_report.txt', class_names=['Normal', 'Abnormal']):
    """
    Generate and save detailed classification report.
    
    Args:
        labels (array): True labels
        predictions (array): Predicted labels
        save_path (str): Path to save report
        class_names (list): Names of classes
    """
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4
    )
    
    print("\n📋 Classification Report:")
    print(report)
    
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(save_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"💾 Saved report to: {save_path}")


def visualize_predictions(
    model,
    test_loader,
    device,
    num_samples=20,
    save_path='results/predictions.png',
    class_names=['Normal', 'Abnormal']
):
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
    
    # Collect some samples
    images_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            if len(images_list) >= num_samples:
                break
            
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            images_list.extend(images.cpu())
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
    
    # Take only num_samples
    images_list = images_list[:num_samples]
    labels_list = labels_list[:num_samples]
    preds_list = preds_list[:num_samples]
    
    # Create grid visualization
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx in range(min(num_samples, rows * cols)):
        ax = axes[idx]
        
        # Denormalize image for display
        img = images_list[idx].permute(1, 2, 0).numpy()
        # Reverse ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Plot image
        ax.imshow(img)
        
        # Title with prediction
        true_label = class_names[labels_list[idx]]
        pred_label = class_names[preds_list[idx]]
        is_correct = labels_list[idx] == preds_list[idx]
        
        color = 'green' if is_correct else 'red'
        title = f"True: {true_label}\nPred: {pred_label}"
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved predictions visualization to: {save_path}")


def analyze_misclassified(
    model,
    test_loader,
    device,
    save_path='results/misclassified.png',
    class_names=['Normal', 'Abnormal'],
    max_samples=16
):
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
    
    # Collect misclassified samples
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    misclassified_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Find misclassified
            misclassified_mask = preds != labels.to(device)
            
            if misclassified_mask.any():
                misc_imgs = images[misclassified_mask].cpu()
                misc_labels = labels[misclassified_mask].cpu().numpy()
                misc_preds = preds[misclassified_mask].cpu().numpy()
                misc_probs = probs[misclassified_mask].cpu().numpy()
                
                misclassified_images.extend(misc_imgs)
                misclassified_labels.extend(misc_labels)
                misclassified_preds.extend(misc_preds)
                misclassified_probs.extend(misc_probs)
            
            if len(misclassified_images) >= max_samples:
                break
    
    if len(misclassified_images) == 0:
        print("✨ No misclassified examples found!")
        return
    
    # Take only max_samples
    num_to_show = min(len(misclassified_images), max_samples)
    misclassified_images = misclassified_images[:num_to_show]
    misclassified_labels = misclassified_labels[:num_to_show]
    misclassified_preds = misclassified_preds[:num_to_show]
    misclassified_probs = misclassified_probs[:num_to_show]
    
    # Create visualization
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx in range(min(num_to_show, rows * cols)):
        ax = axes[idx]
        
        # Denormalize image
        img = misclassified_images[idx].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Plot
        ax.imshow(img)
        
        true_label = class_names[misclassified_labels[idx]]
        pred_label = class_names[misclassified_preds[idx]]
        confidence = misclassified_probs[idx][misclassified_preds[idx]]
        
        title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}"
        ax.set_title(title, color='red', fontsize=9)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_to_show, rows * cols):
        axes[idx].axis('off')
    
    plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold', color='red')
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved misclassified analysis to: {save_path}")
    print(f"   Found {len(misclassified_images)} misclassified examples")


def plot_training_history(history, save_path='results/training_history.png'):
    """
    Plot training history (for federated learning).
    
    Args:
        history (dict): Training history with rounds, losses, accuracies
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = history['rounds']
    
    # Plot 1: Client losses over rounds
    ax1 = axes[0]
    client_losses = history['client_losses']
    
    # Plot each client's losses
    for client_idx in range(len(client_losses[0])):
        client_round_losses = [round_losses[client_idx] for round_losses in client_losses]
        ax1.plot(rounds, client_round_losses, marker='o', label=f'Client {client_idx}')
    
    # Plot average
    avg_losses = [np.mean(round_losses) for round_losses in client_losses]
    ax1.plot(rounds, avg_losses, marker='s', linewidth=2, color='black', label='Average')
    
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Client Training Losses', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation metrics
    ax2 = axes[1]
    ax2.plot(rounds, history['val_accuracy'], marker='o', label='Accuracy', linewidth=2)
    ax2.plot(rounds, history['val_f1'], marker='s', label='F1-Score', linewidth=2)
    
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Global Model Validation Metrics', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved training history to: {save_path}")


def main():
    """Test evaluation utilities."""
    print("Testing evaluation utilities...")
    
    # Create dummy data
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1] * 10)
    predictions = np.array([0, 1, 1, 1, 0, 0, 0, 1] * 10)
    
    # Test confusion matrix
    plot_confusion_matrix(labels, predictions, 'results/test_cm.png')
    
    # Test classification report
    generate_classification_report(labels, predictions, 'results/test_report.txt')
    
    print("\n✨ Evaluation utilities ready!")


if __name__ == "__main__":
    main()
