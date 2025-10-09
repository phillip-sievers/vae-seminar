from dataclasses import dataclass
import torch


@dataclass
class VAEOutput:
    """
    VAE output dataclass
    """

    z_distribution: torch.distributions.MultivariateNormal
    sampled_z: torch.Tensor
    recon_x: torch.Tensor

    loss: torch.Tensor
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor


# add more for other models later
