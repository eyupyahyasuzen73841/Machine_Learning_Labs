import torch
import torch.nn as nn

class EyupVisionTransformer(nn.Module):
    """
    Patch-based Vision Transformer with 49 Spatial Tokens (4x4 patches)
    Image: 28x28 -> Unfold into 7x7=49 patches of shape 4x4=16 pixels
    Linear projection: 16 -> 48
    Self-attention: 2 Transformer Blocks with 4 attention heads
    Head: Token pooling -> Linear(48, 10)
    """
    def __init__(self, patch_size=4, in_channels=1, embed_dim=48, num_heads=4, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2  # 7 * 7 = 49
        patch_dim = in_channels * patch_size * patch_size  # 16
        
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.01)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=96,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        # Extract 4x4 non-overlapping patches
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, 7, 7, 4, 4)
        patches = patches.contiguous().view(B, self.num_patches, -1)  # (B, 49, 16)
        
        tokens = self.patch_proj(patches) + self.pos_embed
        out_tokens = self.transformer(tokens)
        pooled = out_tokens.mean(dim=1)
        return self.classifier(pooled)
