"""
Training loop for standard VAE
================================

This file handles the training loop for VAE experiments.
"""

import torch
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_kl, total_recon = 0.0, 0.0, 0.0
    pbar = tqdm(dataloader, leave=False, desc="Training")

    for batch_idx, (x, _) in enumerate(pbar):
        # x = x.to(device)
        out = model(x)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        optimizer.step()

        # Accumulate losses as tensors (faster than .item() every time)
        total_loss += out.loss.detach()
        total_kl += out.kl_loss.detach()
        total_recon += out.recon_loss.detach()

        # Update progress bar less frequently
        if batch_idx % 10 == 0:
            pbar.set_postfix(
                {
                    "Loss": f"{out.loss.item():.3f}",
                    "KL": f"{out.kl_loss.item():.3f}",
                    "Recon": f"{out.recon_loss.item():.3f}",
                }
            )
    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "kl": total_kl / n,
        "recon": total_recon / n,
    }


@torch.no_grad()
def evaluate_epoch(model, dataloader, device):
    total_loss, total_kl, total_recon = 0.0, 0.0, 0.0

    for x, _ in tqdm(dataloader, leave=False):
        x = x.to(device)
        out = model(x)

        total_loss += out.loss.item()
        total_kl += out.kl_loss.item()
        total_recon += out.recon_loss.item()
    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "kl": total_kl / n,
        "recon": total_recon / n,
    }
