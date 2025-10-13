"""
Training loop for standard VAE
================================

This file handles the training loop for VAE experiments.
"""

import torch
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, scaler=None, log_interval=50):
    """
    Optimized training epoch

    Args:
        log_interval: Print metrics every N batches (0 to disable logging)
    """
    model.train()
    total_loss, total_kl, total_recon = 0.0, 0.0, 0.0
    num_batches = 0

    for batch_idx, (x, _) in enumerate(dataloader):
        optimizer.zero_grad()

        if scaler is not None:
            # l.28 doesnt work on colab?
            # with torch.cuda.amp.autocast(enabled=True):
            out = model(x)
            scaler.scale(out.loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x, track_loss=True)
            out.loss.backward()
            optimizer.step()

        total_loss += out.loss.item()
        total_kl += out.kl_loss.item()
        total_recon += out.recon_loss.item()
        num_batches += 1

        if log_interval > 0 and batch_idx % log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"Batch {batch_idx}/{len(dataloader)}: Loss={avg_loss:.4f}")

    return {
        "loss": total_loss / num_batches,
        "kl": total_kl / num_batches,
        "recon": total_recon / num_batches,
    }


@torch.no_grad()
def evaluate_epoch(model, dataloader, device):
    total_loss, total_kl, total_recon = 0.0, 0.0, 0.0

    for x, _ in tqdm(dataloader, leave=False):
        x = x.to(device)
        out = model(x, track_loss=True)

        total_loss += out.loss.item()
        total_kl += out.kl_loss.item()
        total_recon += out.recon_loss.item()
    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "kl": total_kl / n,
        "recon": total_recon / n,
    }
