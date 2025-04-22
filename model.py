import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Residual Block ---
class ResidualBlock(nn.Module):
    """Residual block with Conv2d -> BN -> ReLU -> Conv2d -> BN layers."""

    def __init__(self, channels: int) -> None:
        """
        Initializes ResidualBlock.
        
        Args:
            channels: number of channels
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: input tensor
        
        Returns:
            output tensor from residual block
        """
        return F.relu(x + self.block(x))


class DepthSTAR(nn.Module):
    """Convolutional-Attention model for depth estimation."""

    def __init__(
        self,
        use_residual_blocks: bool = True,
        use_transformer: bool = True,
        transformer_layers: int = 8,
        transformer_heads: int = 8,
        embed_dim: int = 256,
    ):
        """
        Initializes DepthSTAR.
        
        Args:
            use_residual_blocks: uses residual blocks if True
            use_transformer: uses a transformer bottleneck if True
            transformer_layers: number of transformer encoder layers
            transformer_heads: number of attention heads for the transformer
            embed_dim: embed dimension
        """
        super().__init__()
        self.use_residual_blocks = use_residual_blocks
        self.use_transformer = use_transformer

        # encoder
        encoder_layers = [
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        ]
        if use_residual_blocks:
            encoder_layers.append(ResidualBlock(128))
        encoder_layers += [
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        ]
        if use_residual_blocks:
            encoder_layers.append(ResidualBlock(embed_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # bottleneck
        if use_transformer:
            self.bottleneck = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=transformer_heads,
                    dim_feedforward=embed_dim * 4,
                    batch_first=True,
                ),
                num_layers=transformer_layers,
            )

        # decoder
        decoder_layers = [
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        ]
        if use_residual_blocks:
            decoder_layers.append(ResidualBlock(128))
        decoder_layers += [
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: input tensor
        
        Returns:
            output depth map
        """
        B = x.size(0)
        feat = self.encoder(x)
        if self.use_transformer:
            tokens = feat.flatten(2).transpose(1, 2)
            tokens = self.bottleneck(tokens)
            feat = tokens.transpose(1, 2).reshape(
                B, feat.shape[1], feat.shape[2], feat.shape[3]
            )
        return self.decoder(feat)
