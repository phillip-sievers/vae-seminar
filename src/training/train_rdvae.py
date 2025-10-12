"""
Training loop to apply the RD target Rate fix from the toy model on actual MNIST data
"""

import torch
from src.models.rdvae import RDVAE
from src.utils.datasets import get_bin_mnist_loaders
from src.training.train_epoch import train_epoch, evaluate_epoch
import json
from tqdm import trange
