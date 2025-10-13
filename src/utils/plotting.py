"""
Plotting utility for the experiment
"""

import torch
import matplotlib.pyplot as plt


def visualize_generations(
    models, betas, device, n_samples=8, save_path="./src/results/img/generations.png"
):
    """
    Visualize generations from prior at different beta values
    """

    n_betas = len(betas)
    fig, axes = plt.subplots(n_betas, n_samples, figsize=(16, 2 * n_betas))

    for beta_idx, beta in enumerate(betas):
        model = models[beta]
        model.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, model.latent_dim).to(device)
            x_gen = model.decode(z)

        for i in range(n_samples):
            ax = axes[beta_idx, i] if n_betas > 1 else axes[i]
            ax.imshow(x_gen[i, 0].cpu().numpy(), cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(
                    f"β={beta}",
                    fontsize=12,
                    fontweight="bold",
                    rotation=0,
                    labelpad=40,
                    va="center",
                )

    plt.suptitle(
        "Generations from Prior p(z) = N(0,I)\n",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"saved to {save_path}")
    plt.show()


def visualize_reconstructions(
    models,
    test_loader,
    betas,
    device,
    save_path="./src/results/img/reconstructions.png",
):
    """
    Visualize reconstructions at different beta values
    """

    x, _ = next(iter(test_loader))
    x = x.float()[:8]
    n_betas = len(betas)
    fig, axes = plt.subplots(n_betas + 1, 8, figsize=(16, 2 * (n_betas + 1)))

    # original images
    for i in range(8):
        axes[0, i].imshow(x[i, 0].cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel(
                "Original",
                fontsize=12,
                fontweight="bold",
                rotation=0,
                labelpad=40,
                va="center",
            )

    # reconstructions for each beta
    for beta_idx, beta in enumerate(betas):
        model = models[beta]
        model.eval()

        with torch.no_grad():
            output = model(x, track_loss=False)
            x_recon = output.recon_x

        for i in range(8):
            axes[beta_idx + 1, i].imshow(x_recon[i, 0].cpu().numpy(), cmap="gray")
            axes[beta_idx + 1, i].axis("off")
            if i == 0:
                axes[beta_idx + 1, i].set_ylabel(
                    f"beta={beta}",
                    fontsize=12,
                    fontweight="bold",
                    rotation=0,
                    labelpad=40,
                    va="center",
                )

    plt.suptitle(
        "Reconstructions Across Different beta Values\n",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"saved to {save_path}")
    plt.show()


def plot_rd_curve(results, save_path="./src/results/img/rd_curve.png"):
    """
    Plot Rate-Distortion curve
    """
    betas = sorted(results.keys())
    rates = [results[b]["R"] for b in betas]
    distortions = [results[b]["D"] for b in betas]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Rate-Distortion curve
    ax1 = axes[0]
    ax1.plot(rates, distortions, "o-", linewidth=3, markersize=10, color="#2E86AB")

    # beta values
    for i, beta in enumerate(betas):
        ax1.annotate(
            f"β={beta}",
            (rates[i], distortions[i]),
            textcoords="offset points",
            xytext=(8, -5),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    ax1.set_xlabel("Rate (R) [nats]", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Distortion (D) [nats]", fontsize=13, fontweight="bold")
    ax1.set_title(
        "Rate-Distortion Curve\n(Main Result from Paper)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3, linestyle="--")

    # ELBO vs Rate
    ax2 = axes[1]
    elbos = [-1 * (r + d) for r, d in zip(rates, distortions)]
    ax2.plot(rates, elbos, "s-", linewidth=3, markersize=10, color="#A23B72")

    for i, beta in enumerate(betas):
        ax2.annotate(
            f"β={beta}",
            (rates[i], elbos[i]),
            textcoords="offset points",
            xytext=(8, -5),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )

    ax2.set_xlabel("Rate (R) [nats]", fontsize=13, fontweight="bold")
    ax2.set_ylabel("ELBO = -(R + D) [nats]", fontsize=13, fontweight="bold")
    ax2.set_title(
        "ELBO vs Rate\n(Shows β=1 is not special)", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3, linestyle="--")

    # beta vs (R, D)
    ax3 = axes[2]
    ax3.plot(
        betas, rates, "o-", linewidth=3, markersize=8, label="Rate (R)", color="#F18F01"
    )
    ax3.plot(
        betas,
        distortions,
        "s-",
        linewidth=3,
        markersize=8,
        label="Distortion (D)",
        color="#C73E1D",
    )
    ax3.set_xlabel("β (hyperparameter)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Value [nats]", fontsize=13, fontweight="bold")
    ax3.set_title("Effect of β on R and D", fontsize=14, fontweight="bold")
    ax3.set_xscale("log")
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\ncurve saved to {save_path}")
    plt.show()
