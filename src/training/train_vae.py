"""
Training loop to artificially create a autodecoder according to the paper
"""

import torch

from src.models.vae import VAE
from src.utils.datasets import get_bin_mnist_loaders
from src.training.train_epoch import train_epoch, evaluate_epoch


def train_beta_vae(
    beta,
    latent_dim=64,
    epochs=50,
    lr=1e-3,
    device="cuda",
    data_path="",  # trust the loader lol
    batch_size=1024,
):
    """
    Train a β-VAE with given beta value
    """
    print(f"\nTraining β-VAE with β={beta}")

    train_loader, val_loader, test_loader = get_bin_mnist_loaders(
        batch_size=batch_size, device=device
    )

    model = VAE(latent_dim=latent_dim, beta=beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if device == "cuda":
        scaler = torch.cuda.amp.GradScaler("cuda")

    best_test_loss = float("inf")
    best_D = 0
    best_R = 0

    for epoch in range(epochs):
        train_loss, train_D, train_R = train_epoch(
            model, train_loader, optimizer, scaler
        )
        stats = evaluate_epoch(model, test_loader, device)
        test_loss = stats["loss"]
        test_D = stats["recon"]
        test_R = stats["kl"]
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_D = test_D
            best_R = test_R

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Test Loss: {test_loss:6.2f} | "
                f"D: {test_D:6.2f} | "
                f"R: {test_R:6.2f} | "
                f"ELBO: {-(test_D + test_R):6.2f}"
            )

    print(
        f"\nBest results: D={best_D:.2f}, R={best_R:.2f}, ELBO={-(best_D + best_R):.2f}"
    )

    return best_R, best_D, model
