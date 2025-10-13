import torch
import torch.nn as nn
from .vae import VAE
from .outputs import VAEOutput


class RDVAE(VAE):
    """
    Rate-Distortion VAE with lambda scheduling for controlling the rate-distortion trade-off
    """

    def __init__(
        self,
        latent_dim=64,
        R_target=6.0,
        gamma=100.0,
        abs_loss=True,
        vae_beta=1.0,
        force_collapse=False,
    ):
        super().__init__(latent_dim, beta=vae_beta, force_collapse=force_collapse)
        self.R_target = R_target
        self.gamma = gamma
        self.abs_loss = abs_loss

    def forward(self, x):
        out: VAEOutput = super().forward(x, track_loss=False)

        rate = out.kl_loss
        distortion = out.recon_loss

        if self.abs_loss:
            loss = distortion + abs(self.R_target - rate)
        else:
            loss = distortion + self.gamma * (self.R_target - rate) ** 2

        return VAEOutput(
            z_distribution=out.z_distribution,
            sampled_z=out.sampled_z,
            recon_x=out.recon_x,
            loss=loss,
            recon_loss=distortion,
            kl_loss=rate,
        )
