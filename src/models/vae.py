import torch
import torch.nn as nn
from outputs import VAEOutput


class VAE(nn.Module):
    """
    Variational autoencoder according to Kingma et. al. experiments
    """

    def __init__(self):
        super(VAE, self).__init__()

        # F.1 simple encoder
        self.encoder = nn.Sequential(
            # Normalize to [-1, 1] first
            nn.Conv2d(1, 2 * 32, 5, 1, 2),
            nn.GLU(dim=1),
            nn.Conv2d(32, 2 * 32, 5, 2, 2),
            nn.GLU(dim=1),
            nn.Conv2d(32, 2 * 64, 5, 1, 2),
            nn.GLU(dim=1),
            nn.Conv2d(64, 2 * 64, 5, 2, 2),
            nn.GLU(dim=1),
            nn.Conv2d(64, 2 * 256, 7, 1, 0),
            nn.GLU(dim=1),
            nn.Flatten(),
            nn.Linear(256, 2 * 64),  # double for mean and var
        )
        self.softplus = nn.Softplus()

        # F.2 simple decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 2 * 64, 7, 1, 0),
            nn.GLU(dim=1),
            nn.ConvTranspose2d(64, 2 * 64, 5, 1, 2),
            nn.GLU(dim=1),
            nn.ConvTranspose2d(64, 2 * 64, 5, 2, 2, output_padding=1),
            nn.GLU(dim=1),
            nn.ConvTranspose2d(64, 2 * 32, 5, 1, 2),
            nn.GLU(dim=1),
            nn.ConvTranspose2d(32, 2 * 32, 5, 2, 2, output_padding=1),
            nn.GLU(dim=1),
            nn.ConvTranspose2d(32, 2 * 32, 4, 1, 1),
            nn.GLU(dim=1),
            nn.Conv2d(32, 2 * 1, 5, 1, 2),
            nn.GLU(dim=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """
        Encode given data into the latent space
        """
        features = self.encoder(x)
        mu, logvar = torch.chunk(features, 2, dim=-1)
        scale = self.softplus(logvar)
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, distribution: torch.distributions.MultivariateNormal):
        """
        Use the Reparametrization technique to samplel from Z

        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return distribution.rsample()

    def decode(self, z):
        """
        Decode a latent space sample z
        """
        return self.decoder(z)

    def forward(self, x, track_loss=True):
        """
        Performs a forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            VAEOutput: VAE output dataclass.
        """
        distribution = self.encode(x)
        z = self.reparameterize(distribution)
        reconstructed_x = self.decode(z)

        # compute loss
        recon_loss = (
            nn.functional.binary_cross_entropy(
                reconstructed_x, x + 0.5, reduction="none"
            )
            .sum(-1)
            .mean()
        )
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device)
            .unsqueeze(0)
            .expand(z.shape[0], -1, -1),
        )
        kl_loss = torch.distributions.kl.kl_divergence(distribution, std_normal).mean()

        loss = recon_loss + kl_loss

        return VAEOutput(
            z_distribution=distribution,
            sampled_z=z,
            recon_x=reconstructed_x,
            loss=loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
        )
