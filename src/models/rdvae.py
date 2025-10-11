import torch
import torch.nn as nn
from .vae import VAE
from .outputs import VAEOutput


class RDVAE(VAE):
    """
    Rate-Distortion VAE with lambda scheduling for controlling the rate-distortion trade-off
    """

    def __init__(self, latent_dim=64, R_target=2.0, gamma=100.0):
        super().__init__(latent_dim)
        self.R_target = R_target
        self.gamma = gamma

    def forward(self, x, track_loss=True):
        out: VAEOutput = super().forward(x, False)

        rate = out.kl_loss
        distortion = out.recon_loss

        # programmatic switch
        loss = distortion + abs(self.R_target - rate.detach())
        # loss = distortion + self.gamma * (self.R_target - rate)**2

        return VAEOutput(
            z_distribution=out.z_distribution,
            sampled_z=out.sampled_z,
            recon_x=out.recon_x,
            loss=loss,
            recon_loss=distortion,
            kl_loss=rate,
        )
