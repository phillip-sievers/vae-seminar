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

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    num_batches = 0

    for x, _ in tqdm(dataloader, leave=False, desc="Training"):
        # x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast("cuda"):
                out = model(x)
            scaler.scale(out.loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            out.loss.backward()
            optimizer.step()

        total_loss += out.loss.detach()
        total_kl += out.kl_loss.detach()
        total_recon += out.recon_loss.detach()
        num_batches += 1

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
