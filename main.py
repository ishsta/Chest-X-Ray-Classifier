"""
Main execution script for Chest X-Ray Classification

Supports:
- Data loading and preparation
- Centralized training
- Federated learning with FedAvg
- Model evaluation
"""

import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
import pandas as pd
import pickle

# Import project modules
from models import create_vit_small, create_vit_base, create_lstm_small, create_lstm_base
from dataset import ChestXrayDataset, create_dataloaders
from train import get_device, create_optimizer, create_scheduler, save_checkpoint
from federated import partition_data_for_clients, create_client_dataloaders, train_federated
from evaluate import (
    evaluate_on_test_set,
    plot_confusion_matrix,
    generate_classification_report,
    visualize_predictions,
    analyze_misclassified,
    plot_training_history
)


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
    
    # Create datasets
    from dataset import get_train_transform, get_val_transform
    
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
    
    val_loader = torch.utils.data.DataLoader(
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
    
    test_loader = torch.utils.data.DataLoader(
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
