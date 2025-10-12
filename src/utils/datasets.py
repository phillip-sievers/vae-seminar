"""
Dataset utilities and data loaders
"""

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict
import os

import urllib.request
import ssl
import shutil
from .directory_structure import get_repo_root


class BinaryMNIST(Dataset):
    """
    Static binary MNIST according to Paper
    """

    def __init__(self, data: str = "/src/data/", split="train"):
        assert split in {"train", "valid", "test"}
        base = get_repo_root()
        self.root = os.path.join(base, data)
        print(self.root)
        self.split = split

        self._download(split)

        file_path = os.path.join(self.root, f"binarized_mnist_{split}.amat")
        data = np.loadtxt(file_path)
        self.data = torch.tensor(data, dtype=torch.float32).view(-1, 1, 28, 28)

    def _download(self, split="train"):
        """
        Doanload the data if needed
        """
        file_names = {
            "train": "binarized_mnist_train.amat",
            "valid": "binarized_mnist_valid.amat",
            "test": "binarized_mnist_test.amat",
        }
        file_name = file_names[split]
        print(f"Checking if {file_name} exists")
        path = os.path.join(self.root, file_name)
        if not os.path.exists(path):
            print(f"Downloading {file_name}")
            context = ssl._create_unverified_context()
            url = f"http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/{file_name}"
            with urllib.request.urlopen(url, context=context) as response:
                with open(path, "wb") as file:
                    shutil.copyfileobj(response, file)
            print("Succesfully downloaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x, 0


def get_bin_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = "./src/data/",
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get MNIST train, validation, and test dataloaders.x

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_ds = BinaryMNIST(data_dir, "train")
    val_ds = BinaryMNIST(data_dir, "valid")
    test_ds = BinaryMNIST(data_dir, "test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# to be cont. for visualizer and stuff


# ============================================================
# Visualization
# ============================================================
def visualize_batch(
    dataloader: DataLoader, num_samples: int = 8, title: str = "Data Samples"
):
    """
    Visualize a batch of images from dataloader.
    """
    imgs, _ = next(iter(dataloader))
    imgs = imgs[:num_samples]

    ncols = min(num_samples, 8)
    nrows = int(np.ceil(num_samples / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i >= num_samples:
            ax.axis("off")
            continue
        img = imgs[i]
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(0), cmap="gray")
        else:
            ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Statistics computation
# ============================================================
def compute_dataset_stats(dataloader: DataLoader) -> Dict[str, torch.Tensor]:
    """
    Compute mean and std statistics for a dataset.
    """
    mean = 0.0
    std = 0.0
    total = 0
    for x, _ in dataloader:
        batch_samples = x.size(0)
        x = x.view(batch_samples, x.size(1), -1)
        mean += x.mean(2).sum(0)
        std += x.std(2).sum(0)
        total += batch_samples
    mean /= total
    std /= total
    return {"mean": mean, "std": std}
