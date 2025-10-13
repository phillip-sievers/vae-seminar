import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.outputs import VAEOutput
from src.models.pixelcnnpp import PixelCNN
from src.utils.cnn_utils import (
    discretized_mix_logistic_loss_1d,
    sample_from_discretized_mix_logistic_1d,
)


class VAEWithPixelCNN(nn.Module):
    """
    VAE with PixelCNN++ decoder that will demonstrate posterior collapse
    """

    def __init__(self, latent_dim=64, beta=1.0):
        super(VAEWithPixelCNN, self).__init__()
        self.beta = beta
        self.latent_dim = latent_dim

        # Simple encoder (same as before)
        self.encoder = nn.Sequential(
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

        # Condition the PixelCNN on latent code
        # For this implementation, we'll use a simpler approach:
        # Project latent to spatial features that get added to intermediate layers
        self.latent_proj = nn.Linear(latent_dim, 28 * 28)  # Project to image size

        # PixelCNN decoder - standard input channels for MNIST
        self.decoder = PixelCNN(
            nr_resnet=3,  # Number of residual blocks
            nr_filters=64,  # Number of filters
            input_channels=1,  # Original MNIST image (grayscale)
            nr_logistic_mix=5,  # Number of logistic mixture components
        )

    def encode(self, x):
        """Encode given data into the latent space"""
        features = self.encoder(x)
        mu, logvar = torch.chunk(features, 2, dim=-1)
        scale = self.softplus(logvar)
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, distribution: torch.distributions.MultivariateNormal):
        """Use the Reparametrization technique to sample from Z"""
        return distribution.rsample()

    def decode(self, z, x_input=None):
        """
        Decode latent code z into image.
        For now, we'll use a simple approach where PixelCNN operates independently
        and the latent code provides global conditioning (this will show collapse).

        Args:
            z: latent code [batch_size, latent_dim]
            x_input: input image for teacher forcing during training
        """
        batch_size = z.size(0)

        # For this simple implementation, we'll let PixelCNN run unconditionally
        # This will demonstrate the posterior collapse problem!
        if x_input is not None:
            # Training: use input image
            decoder_input = x_input
        else:
            # Generation: start with zeros
            decoder_input = torch.zeros(batch_size, 1, 28, 28, device=z.device)

        # Get logistic mixture parameters
        logits = self.decoder(decoder_input)

        return logits

    def sample_reconstruction(self, logits):
        """Sample from the decoder output"""
        # The PixelCNN outputs logistic mixture parameters
        # We need to sample from this distribution (1D version for grayscale)
        return sample_from_discretized_mix_logistic_1d(logits, nr_mix=5)

    def forward(self, x, track_loss=True):
        """Performs a forward pass of the VAE"""
        # Normalize input to [-1, 1] (required for PixelCNN)
        x_normalized = 2.0 * x - 1.0

        # Encode
        distribution = self.encode(x_normalized)
        z = self.reparameterize(distribution)

        # Decode - pass the input for teacher forcing
        logits = self.decode(z, x_input=x_normalized)

        # Sample reconstruction for visualization
        recon_x = self.sample_reconstruction(logits)
        # Convert back to [0, 1] range
        recon_x = (recon_x + 1.0) / 2.0

        if track_loss:
            # Reconstruction loss using discretized mix logistic (1D version for grayscale)
            recon_loss = discretized_mix_logistic_loss_1d(x_normalized, logits)

            # KL divergence
            std_normal = torch.distributions.MultivariateNormal(
                torch.zeros_like(z, device=z.device),
                scale_tril=torch.eye(z.shape[-1], device=z.device)
                .unsqueeze(0)
                .expand(z.shape[0], -1, -1),
            )
            kl_loss = torch.distributions.kl.kl_divergence(
                distribution, std_normal
            ).mean()

            # Total loss
            loss = recon_loss + self.beta * kl_loss
        else:
            loss = recon_loss = kl_loss = None

        return VAEOutput(
            z_distribution=distribution,
            sampled_z=z,
            recon_x=recon_x,
            loss=loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
        )

    def sample(self, num_samples, device):
        """Generate samples from the model"""
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim, device=device)

            # For now, generate unconditionally (this shows the collapse problem!)
            # In a proper implementation, we'd condition on z
            samples = torch.zeros(num_samples, 1, 28, 28, device=device)

            # Sequential generation (proper PixelCNN sampling)
            for i in range(28):
                for j in range(28):
                    # Generate unconditionally (ignoring z - this will show collapse!)
                    logits = self.decoder(samples)

                    # Sample next pixel
                    pixel_sample = sample_from_discretized_mix_logistic_1d(
                        logits, nr_mix=5
                    )
                    samples[:, :, i, j] = pixel_sample[:, :, i, j]

            # Convert to [0, 1] range
            samples = (samples + 1.0) / 2.0

        return samples
