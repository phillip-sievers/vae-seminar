#!/usr/bin/env python3
"""
Debug script to understand the shape issues
"""

import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.models.vae_pixelcnn import VAEWithPixelCNN


def debug_shapes():
    print("Debugging shape issues...")

    device = torch.device("cpu")
    model = VAEWithPixelCNN(latent_dim=64, beta=1.0).to(device)

    # Create input
    batch_size = 2
    x = torch.rand(batch_size, 1, 28, 28, device=device)
    print(f"Input shape: {x.shape}")

    # Normalize input
    x_normalized = 2.0 * x - 1.0
    print(f"Normalized input shape: {x_normalized.shape}")

    # Test encoding
    distribution = model.encode(x_normalized)
    z = model.reparameterize(distribution)
    print(f"Latent z shape: {z.shape}")

    # Test decoder
    logits = model.decode(z, x_input=x_normalized)
    print(f"Decoder output shape: {logits.shape}")

    # Check what the PixelCNN expects
    print(f"PixelCNN nr_logistic_mix: {model.decoder.nr_logistic_mix}")
    print(f"PixelCNN input_channels: {model.decoder.input_channels}")

    # Expected output channels for grayscale with 5 mixtures should be:
    # 3 * nr_logistic_mix = 3 * 5 = 15
    expected_channels = 3 * model.decoder.nr_logistic_mix
    print(f"Expected output channels: {expected_channels}")


if __name__ == "__main__":
    debug_shapes()
