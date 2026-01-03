"""
Federated Learning package for distributed training.
"""

from .fed_avg import (
    partition_data_for_clients,
    create_client_dataloaders,
    federated_averaging,
    train_federated,
    get_model_weights,
    set_model_weights
)

__all__ = [
    'partition_data_for_clients',
    'create_client_dataloaders',
    'federated_averaging',
    'train_federated',
    'get_model_weights',
    'set_model_weights',
]
