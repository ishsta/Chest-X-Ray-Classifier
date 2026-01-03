"""
Federated Learning Implementation using FedAvg

Implements:
- Client data partitioning
- Federated Averaging (FedAvg) algorithm
- Client training coordination
- Global model aggregation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm


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
    
    if allocation == 'balanced':
        # Equal distribution
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, num_clients)
        client_data = {f'client_{i}': split.tolist() for i, split in enumerate(split_indices)}
    
    elif allocation == 'random':
        # Random sizes (but ensuring each client has data)
        min_samples = len(indices) // (num_clients * 2)
        client_data = {}
        remaining = indices.copy()
        
        for i in range(num_clients - 1):
            size = np.random.randint(min_samples, len(remaining) - min_samples * (num_clients - i - 1))
            np.random.shuffle(remaining)
            client_data[f'client_{i}'] = remaining[:size].tolist()
            remaining = remaining[size:]
        
        client_data[f'client_{num_clients-1}'] = remaining.tolist()
    
    elif allocation == 'iid':
        # Independent and identically distributed
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, num_clients)
        client_data = {f'client_{i}': split.tolist() for i, split in enumerate(split_indices)}
    
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
    model.load_state_dict(weights)


def federated_averaging(client_weights, client_sizes):
    """
    Aggregate client model weights using Federated Averaging.
    
    Args:
        client_weights (list): List of client state_dicts
        client_sizes (list): List of dataset sizes for each client
        
    Returns:
        dict: Averaged state dictionary
    """
    # Calculate weights based on dataset sizes
    total_size = sum(client_sizes)
    weights_ratios = [size / total_size for size in client_sizes]
    
    # Initialize averaged weights
    avg_weights = {}
    
    # Get all parameter names from first client
    param_names = client_weights[0].keys()
    
    for param_name in param_names:
        # Weighted average of each parameter
        avg_weights[param_name] = sum(
            client_weights[i][param_name] * weights_ratios[i]
            for i in range(len(client_weights))
        )
    
    return avg_weights


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
    total_samples = 0
    
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
            
            epoch_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
        
        total_loss += epoch_loss
    
    avg_loss = total_loss / (total_samples * local_epochs)
    
    return {
        'client_id': client_id,
        'loss': avg_loss,
        'samples': len(dataloader.dataset)
    }


def train_federated(
    global_model,
    client_loaders,
    val_loader,
    num_rounds=5,
    local_epochs=2,
    lr=1e-4,
    device='cuda',
    save_dir='checkpoints'
):
    """
    Execute federated learning training with FedAvg.
    
    Args:
        global_model (nn.Module): Initial global model
        client_loaders (dict): Client ID -> DataLoader mapping
        val_loader (DataLoader): Validation data loader
        num_rounds (int): Number of communication rounds
        local_epochs (int): Local epochs per round
        lr (float): Learning rate
        device: Device to train on
        save_dir (str): Directory to save checkpoints
        
    Returns:
        dict: Training history
    """
    from train import evaluate_model, create_optimizer
    
    print("=" * 60)
    print("Federated Learning Training (FedAvg)")
    print("=" * 60)
    print(f"Clients: {len(client_loaders)}")
    print(f"Communication rounds: {num_rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Device: {device}")
    print("=" * 60)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize history
    history = {
        'rounds': [],
        'client_losses': [],
        'val_accuracy': [],
        'val_loss': [],
        'val_f1': []
    }
    
    criterion = nn.CrossEntropyLoss()
    
    # Global model to device
    global_model = global_model.to(device)
    
    # Federated training loop
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'='*60}")
        
        # Store client weights and sizes
        client_weights_list = []
        client_sizes = []
        client_losses = []
        
        # Train each client
        for client_id, client_loader in client_loaders.items():
            print(f"\n📱 Training {client_id} ({len(client_loader.dataset)} samples)...")
            
            # Create local model (copy of global)
            local_model = deepcopy(global_model)
            local_model = local_model.to(device)
            
            # Create optimizer for local training
            optimizer = create_optimizer(local_model, 'adam', lr=lr)
            
            # Train locally
            client_metrics = train_client(
                client_id,
                local_model,
                client_loader,
                criterion,
                optimizer,
                device,
                local_epochs
            )
            
            # Collect weights and metrics
            client_weights_list.append(get_model_weights(local_model))
            client_sizes.append(client_metrics['samples'])
            client_losses.append(client_metrics['loss'])
            
            print(f"   Loss: {client_metrics['loss']:.4f}")
        
        # Aggregate weights using FedAvg
        print(f"\n🔄 Aggregating client models...")
        global_weights = federated_averaging(client_weights_list, client_sizes)
        set_model_weights(global_model, global_weights)
        
        # Evaluate global model
        print(f"\n📊 Evaluating global model...")
        val_metrics = evaluate_model(
            global_model,
            val_loader,
            criterion,
            device,
            desc=f"Round {round_num} Validation"
        )
        
        # Update history
        history['rounds'].append(round_num)
        history['client_losses'].append(client_losses)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Print round summary
        print(f"\n📈 Round {round_num} Summary:")
        print(f"   Avg Client Loss: {np.mean(client_losses):.4f}")
        print(f"   Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"   Val F1-Score: {val_metrics['f1']:.4f}")
        print(f"   Val Loss: {val_metrics['loss']:.4f}")
        
        # Save checkpoint
        checkpoint_path = save_dir / f"federated_round_{round_num}.pth"
        torch.save({
            'round': round_num,
            'model_state_dict': global_model.state_dict(),
            'val_metrics': val_metrics,
            'history': history
        }, checkpoint_path)
        print(f"\n💾 Saved checkpoint: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print("✨ Federated training complete!")
    print(f"{'='*60}")
    
    return history


def main():
    """Test federated learning module."""
    print("Testing federated learning utilities...")
    
    # Test data partitioning
    indices = list(range(1000))
    client_data = partition_data_for_clients(indices, num_clients=3, allocation='balanced')
    
    print(f"\n✅ Client data partitioning:")
    for client_id, client_indices in client_data.items():
        print(f"   {client_id}: {len(client_indices)} samples")
    
    # Test weight averaging
    print(f"\n✅ Testing federated averaging...")
    dummy_weights = [
        {'weight': torch.tensor([1.0, 2.0, 3.0])},
        {'weight': torch.tensor([2.0, 3.0, 4.0])},
        {'weight': torch.tensor([3.0, 4.0, 5.0])}
    ]
    sizes = [100, 200, 300]
    avg = federated_averaging(dummy_weights, sizes)
    print(f"   Averaged weight: {avg['weight']}")
    expected = torch.tensor([2.333, 3.333, 4.333])
    print(f"   Expected: {expected}")
    
    print("\n✨ Federated learning utilities ready!")


if __name__ == "__main__":
    main()
