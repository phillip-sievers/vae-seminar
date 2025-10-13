"""
Custom decoder architecture based on the paper specification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearGatedConv2d(nn.Module):
    """Linearly gated convolutional layer"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels * 2, kernel_size, stride, padding
        )

    def forward(self, x):
        x = self.conv(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        return x1 * torch.sigmoid(x2)


class LinearGatedLinear(nn.Module):
    """Linearly gated linear layer"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x):
        x = self.linear(x)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return x1 * torch.sigmoid(x2)


class ResNetBlock(nn.Module):
    """ResNet block with linear gated activations"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = LinearGatedConv2d(channels, channels)
        self.conv2 = LinearGatedConv2d(channels, channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class CNNPPDecoder(nn.Module):
    """
    According to paper
    Decoder architecture as specified in the paper:
    - 6 blocks with sizes: (28×28) > (14×14) > (7×7) > (7×7) > (14×14) > (28×28)
    - 2 ResNet layers per block
    - Feature depth of 64 throughout
    - Shortcut connections between matching sized feature maps
    """

    def __init__(self, latent_dim=64):
        super().__init__()

        # Dense linearly gated layer: latent_dim > 784
        self.latent_to_image = LinearGatedLinear(latent_dim, 784)

        # Initial conv to get to 64 channels
        # Input: (28×28×2) > Output: (28×28×64)
        self.initial_conv = LinearGatedConv2d(2, 64)

        # Block 1: (28×28) - 2 ResNet layers
        self.block1_res1 = ResNetBlock(64)
        self.block1_res2 = ResNetBlock(64)

        # Downsample: (28×28) > (14×14)
        self.downsample1 = LinearGatedConv2d(64, 64, stride=2)

        # Block 2: (14×14) - 2 ResNet layers
        self.block2_res1 = ResNetBlock(64)
        self.block2_res2 = ResNetBlock(64)

        # Downsample: (14×14) > (7×7)
        self.downsample2 = LinearGatedConv2d(64, 64, stride=2)

        # Block 3: (7×7) - 2 ResNet layers (bottleneck)
        self.block3_res1 = ResNetBlock(64)
        self.block3_res2 = ResNetBlock(64)

        # Block 4: (7×7) - 2 ResNet layers (bottleneck)
        self.block4_res1 = ResNetBlock(64)
        self.block4_res2 = ResNetBlock(64)

        # Upsample: (7×7) > (14×14)
        self.upsample1 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # Block 5: (14×14) - 2 ResNet layers
        self.block5_res1 = ResNetBlock(64)
        self.block5_res2 = ResNetBlock(64)

        # Upsample: (14×14) > (28×28)
        self.upsample2 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # Block 6: (28×28) - 2 ResNet layers
        self.block6_res1 = ResNetBlock(64)
        self.block6_res2 = ResNetBlock(64)

        # Final 1×1 conv: (28×28×64) > (28×28×1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, z, target_image):
        """
        Args:
            z: Latent code (batch_size, latent_dim)
            target_image: Target image (batch_size, 1, 28, 28)
        """
        # Dense layer: latent > 784 > reshape to (28×28×1)
        z_image = self.latent_to_image(z)
        z_image = z_image.view(-1, 1, 28, 28)

        # Concatenate with target image: (28×28×1) + (28×28×1) > (28×28×2)
        x = torch.cat([z_image, target_image], dim=1)
        # Initial conv to 64 channels
        x = self.initial_conv(x)

        # Block 1: (28×28)
        skip_28x28 = x
        x = self.block1_res1(x)
        x = self.block1_res2(x)

        # Downsample to (14×14)
        x = self.downsample1(x)

        # Block 2: (14×14)
        skip_14x14 = x
        x = self.block2_res1(x)
        x = self.block2_res2(x)

        # Downsample to (7×7)
        x = self.downsample2(x)

        # Block 3: (7×7) - bottleneck
        x = self.block3_res1(x)
        x = self.block3_res2(x)

        # Block 4: (7×7) - bottleneck
        x = self.block4_res1(x)
        x = self.block4_res2(x)

        # Upsample to (14×14)
        x = self.upsample1(x)

        # Block 5: (14×14) with shortcut connection
        x = x + skip_14x14
        x = self.block5_res1(x)
        x = self.block5_res2(x)

        # Upsample to (28×28)
        x = self.upsample2(x)

        # Block 6: (28×28) with shortcut connection
        x = x + skip_28x28
        x = self.block6_res1(x)
        x = self.block6_res2(x)

        # Final 1×1 conv to get logits
        logits = self.final_conv(x)

        return torch.sigmoid(logits)
