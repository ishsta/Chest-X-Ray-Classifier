"""
Chest X-Ray Classification Models

Available architectures:
- Vision Transformer (ViT)
- LSTM-based Classifier

Factory functions:
- create_vit_small, create_vit_base
- create_lstm_small, create_lstm_base
"""

import torch
import torch.nn as nn
import math


# ============================================================================
# VISION TRANSFORMER (ViT) IMPLEMENTATION
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Convert image into patches and embed them.
    
    Args:
        img_size (int): Input image size (assuming square)
        patch_size (int): Size of each patch
        in_channels (int): Number of input channels (3 for RGB)
        embed_dim (int): Embedding dimension
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Conv2d with kernel_size=patch_size and stride=patch_size
        # effectively divides image into non-overlapping patches
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Tensor of shape (B, num_patches, embed_dim)
        """
        # x: (B, 3, 224, 224) -> (B, embed_dim, 14, 14)
        x = self.projection(x)
        
        # (B, embed_dim, 14, 14) -> (B, embed_dim, 196) -> (B, 196, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input of shape (B, N, embed_dim)
        
        Returns:
            Output of shape (B, N, embed_dim)
        """
        B, N, C = x.shape
        
        # (B, N, 3*embed_dim) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Attention: (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = attn @ v
        
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, embed_dim)
        x = x.transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network).
    
    Args:
        embed_dim (int): Input/output dimension
        hidden_dim (int): Hidden layer dimension
        dropout (float): Dropout rate
    """
    
    def __init__(self, embed_dim=768, hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block: Self-Attention + MLP with residual connections.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        dropout (float): Dropout rate
    """
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class ViTClassifier(nn.Module):
    """
    Vision Transformer for binary classification.
    
    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        embed_dim (int): Embedding dimension
        num_layers (int): Number of transformer encoder blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP hidden dimension ratio
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        B = x.shape[0]
        
        # Patch embedding: (B, 3, 224, 224) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Prepend class token: (B, num_patches, embed_dim) -> (B, num_patches+1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through transformer encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Extract class token and classify
        cls_token_final = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_token_final)  # (B, num_classes)
        
        return logits


# ============================================================================
# LSTM CLASSIFIER IMPLEMENTATION
# ============================================================================

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


# ============================================================================
# MODEL FACTORY FUNCTIONS
# ============================================================================

def create_vit_small(num_classes=2, img_size=224):
    """Create a smaller ViT model for faster training."""
    return ViTClassifier(
        img_size=img_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=384,
        num_layers=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1
    )


def create_vit_base(num_classes=2, img_size=224):
    """Create the base ViT model."""
    return ViTClassifier(
        img_size=img_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    )


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


# ============================================================================
# TESTING
# ============================================================================

def main():
    """Test all models."""
    print("Testing Chest X-Ray Classification Models")
    print("=" * 60)
    
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Test ViT Small
    print("\n1. Testing ViT Small:")
    vit_model = create_vit_small(num_classes=2)
    num_params = sum(p.numel() for p in vit_model.parameters())
    print(f"   Parameters: {num_params:,}")
    print(f"   Input shape: {x.shape}")
    
    with torch.no_grad():
        output = vit_model(x)
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, 2)
    print("   ✅ ViT Small test passed!")
    
    # Test LSTM
    print("\n2. Testing LSTM Classifier:")
    lstm_model = create_lstm_small(num_classes=2)
    num_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    with torch.no_grad():
        output = lstm_model(x)
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, 2)
    print("   ✅ LSTM test passed!")
    
    # Test LSTM with Attention
    print("\n3. Testing LSTM with Attention:")
    lstm_attn_model = create_lstm_small(num_classes=2, use_attention=True)
    num_params = sum(p.numel() for p in lstm_attn_model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    with torch.no_grad():
        output = lstm_attn_model(x)
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, 2)
    print("   ✅ LSTM with Attention test passed!")
    
    print("\n" + "=" * 60)
    print("✨ All model tests passed!")


if __name__ == "__main__":
    main()
