"""
LSTM-based Classifier for Chest X-Ray Classification

Treats image patches as a sequence and processes with LSTM layers.
Designed for binary classification (Normal vs Abnormal).
"""

import torch
import torch.nn as nn


class PatchSequenceExtractor(nn.Module):
    """
    Extract patches from image and treat them as a sequence.
    
    Args:
        img_size (int): Input image size (assuming square)
        patch_size (int): Size of each patch
        in_channels (int): Number of input channels (3 for RGB)
        embed_dim (int): Embedding dimension for each patch
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer to extract and embed patches
        self.patch_extractor = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(embed_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Tensor of shape (B, num_patches, embed_dim)
        """
        # x: (B, 3, 224, 224) -> (B, embed_dim, 14, 14)
        x = self.patch_extractor(x)
        x = self.bn(x)
        x = self.activation(x)
        
        # (B, embed_dim, 14, 14) -> (B, embed_dim, 196) -> (B, 196, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for chest X-ray images.
    
    Architecture:
    1. Extract image patches as sequences
    2. Process with bidirectional LSTM layers
    3. Use final hidden state for classification
    
    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        embed_dim (int): Patch embedding dimension
        hidden_dim (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        bidirectional (bool): Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Patch extraction
        self.patch_extractor = PatchSequenceExtractor(
            img_size, patch_size, in_channels, embed_dim
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        # If bidirectional, hidden_dim is doubled
        classifier_input_dim = hidden_dim * self.num_directions
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        B = x.shape[0]
        
        # Extract patches as sequence: (B, 3, 224, 224) -> (B, num_patches, embed_dim)
        patches = self.patch_extractor(x)
        
        # Pass through LSTM
        # lstm_out: (B, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, B, hidden_dim)
        # c_n: (num_layers * num_directions, B, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(patches)
        
        # Use the final hidden state for classification
        # Take last layer's hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # h_n[-2]: forward last layer, h_n[-1]: backward last layer
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, hidden_dim * 2)
        else:
            final_hidden = h_n[-1]  # (B, hidden_dim)
        
        # Apply dropout
        final_hidden = self.dropout(final_hidden)
        
        # Classify
        logits = self.classifier(final_hidden)  # (B, num_classes)
        
        return logits


class LSTMClassifierWithAttention(nn.Module):
    """
    Enhanced LSTM classifier with attention mechanism.
    
    Uses attention to weight the LSTM outputs before classification.
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        
        # Patch extraction
        self.patch_extractor = PatchSequenceExtractor(
            img_size, patch_size, in_channels, embed_dim
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_dim * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        # Extract patches as sequence
        patches = self.patch_extractor(x)  # (B, num_patches, embed_dim)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(patches)  # (B, num_patches, hidden_dim * num_directions)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # (B, num_patches, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize across sequence
        
        # Apply attention: weighted sum of LSTM outputs
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, hidden_dim * num_directions)
        
        # Apply dropout
        context = self.dropout(context)
        
        # Classify
        logits = self.classifier(context)  # (B, num_classes)
        
        return logits


def create_lstm_small(num_classes=2, img_size=224, use_attention=False):
    """Create a smaller LSTM model for faster training."""
    if use_attention:
        return LSTMClassifierWithAttention(
            img_size=img_size,
            patch_size=16,
            num_classes=num_classes,
            embed_dim=128,
            hidden_dim=128,
            num_layers=1,
            dropout=0.2,
            bidirectional=True
        )
    else:
        return LSTMClassifier(
            img_size=img_size,
            patch_size=16,
            num_classes=num_classes,
            embed_dim=128,
            hidden_dim=128,
            num_layers=1,
            dropout=0.2,
            bidirectional=True
        )


def create_lstm_base(num_classes=2, img_size=224, use_attention=False):
    """Create the base LSTM model."""
    if use_attention:
        return LSTMClassifierWithAttention(
            img_size=img_size,
            patch_size=16,
            num_classes=num_classes,
            embed_dim=256,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )
    else:
        return LSTMClassifier(
            img_size=img_size,
            patch_size=16,
            num_classes=num_classes,
            embed_dim=256,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )


def main():
    """Test LSTM models."""
    print("Testing LSTM Classifiers...")
    
    # Test basic LSTM
    print("\n1. Testing Basic LSTM Classifier:")
    model = create_lstm_small(num_classes=2)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    print(f"   Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, 2), f"Output shape mismatch: {output.shape}"
    print("   ✅ Basic LSTM test passed!")
    
    # Test LSTM with attention
    print("\n2. Testing LSTM with Attention:")
    model_attn = create_lstm_small(num_classes=2, use_attention=True)
    num_params = sum(p.numel() for p in model_attn.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    with torch.no_grad():
        output = model_attn(x)
    
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, 2), f"Output shape mismatch: {output.shape}"
    print("   ✅ LSTM with Attention test passed!")
    
    print("\n✅ All LSTM model tests passed!")


if __name__ == "__main__":
    main()
