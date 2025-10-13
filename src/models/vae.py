import torch
import torch.nn as nn
from src.models.outputs import VAEOutput
from src.models.auto_decoder_paper import AutoDecoder, CNNPPDecoder


class VAE(nn.Module):
    """
    Variational autoencoder according to Kingma et. al. experiments
    """

    def __init__(self, latent_dim=64, beta=1.0, force_collapse=False):
        super(VAE, self).__init__()
        self.beta = beta
        self.latent_dim = latent_dim
        self.force_collapse = force_collapse

        # F.1 simple encoder
        if force_collapse:
            # Weaker encoder to make posterior collapse more likely
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 5, 2, 2),  # Reduced capacity
                nn.ReLU(),  # Use ReLU instead of GLU
                nn.Conv2d(16, 32, 5, 2, 2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 7, 1, 0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64, 2 * latent_dim),  # Simple linear layer
            )
        else:
            # Standard encoder
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
                nn.Linear(256, 2 * latent_dim),  # double for mean and var
            )
        self.softplus = nn.Softplus()

        # Paper decoder handles latent-to-image mapping internally
        # F.2 simple decoder (commented out)
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 2 * 64, 7, 1, 0),
        #     nn.GLU(dim=1),
        #     nn.ConvTranspose2d(64, 2 * 64, 5, 1, 2),
        #     nn.GLU(dim=1),
        #     nn.ConvTranspose2d(64, 2 * 64, 5, 2, 2, output_padding=1),
        #     nn.GLU(dim=1),
        #     nn.ConvTranspose2d(64, 2 * 32, 5, 1, 2),
        #     nn.GLU(dim=1),
        #     nn.ConvTranspose2d(32, 2 * 32, 5, 2, 2, output_padding=1),
        #     nn.GLU(dim=1),
        #     nn.ConvTranspose2d(32, 2 * 32, 5, 1, 2),
        #     nn.GLU(dim=1),
        #     nn.Conv2d(32, 2 * 1, 5, 1, 2),
        #     nn.GLU(dim=1),
        #     nn.Sigmoid(),
        # )
        # Using autodecoder (generates from latent only):
        self.decoder = AutoDecoder(latent_dim=latent_dim)

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
        Decode a latent space sample z using autodecoder

        Args:
            z: Latent code (batch_size, latent_dim)
        Returns:
            Generated image (batch_size, 1, 28, 28)
        """
        # Autodecoder generates images from latent codes only
        return self.decoder(z)

    def forward(self, x, track_loss=True):
        """
        Performs a forward pass of the VAE.
        """
        distribution = self.encode(x)
        z = self.reparameterize(distribution)
        # Autodecoder only needs latent code
        reconstructed_x = self.decode(z)

        recon_loss = nn.functional.binary_cross_entropy(
            reconstructed_x, x, reduction="sum"
        ) / x.size(0)

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device)
            .unsqueeze(0)
            .expand(z.shape[0], -1, -1),
        )
        kl_loss = torch.distributions.kl.kl_divergence(distribution, std_normal).mean()

        if track_loss:
            # Apply different beta strategies for collapse demonstration
            if self.force_collapse:
                # High beta or annealing strategy that causes collapse
                effective_beta = self.beta * 10.0  # Much higher beta
            else:
                effective_beta = self.beta

            loss = recon_loss + effective_beta * kl_loss
        else:
            loss = None

        return VAEOutput(
            z_distribution=distribution,
            sampled_z=z,
            recon_x=reconstructed_x,
            loss=loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
        )
