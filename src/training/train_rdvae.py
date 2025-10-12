"""
Training loop for RD-VAE with lambda scheduling
===============================================

This file handles the training loop for RD-VAE experiments with dynamic lambda scheduling.
"""

import torch
from src.models.rdvae import RDVAE
from src.utils.datasets import get_bin_mnist_loaders
from src.training.train_epoch import train_epoch, evaluate_epoch
import json
from tqdm import trange

device = "mps" if torch.backends.mps.is_available() else "cpu"

train_loader, val_loader, test_loader = get_bin_mnist_loaders(batch_size=128)

R_values = [0.5, 1.0, 2.0, 4.0, 8.0]
results = []

for R_target in R_values:
    print(f"\n=== Training RDVAE with R_target={R_target} ===")
    model = RDVAE(latent_dim=64, R_target=R_target, gamma=100.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in trange(20, desc=f"R={R_target}"):
        train_stats = train_epoch(model, train_loader, optimizer, device)
        if epoch % 5 == 0:
            print(
                f"[{epoch:03d}] train_loss={train_stats['loss']:.2f} KL={train_stats['kl']:.2f} recon={train_stats['recon']:.2f}"
            )

    test_stats = evaluate_epoch(model, test_loader, device)
    test_stats["R_target"] = R_target
    results.append(test_stats)

with open("rd_results.json", "w") as f:
    json.dump(results, f, indent=4)
