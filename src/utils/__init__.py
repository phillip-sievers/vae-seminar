"""
Utils module initialization
===========================

This file makes the utils directory a Python package and provides convenient imports.
"""

from .datasets import (
    get_bin_mnist_loaders,
    visualize_batch,
    compute_dataset_stats,
)
from .metrics import (
    compute_rate,
    compute_distortion,
    compute_elbo,
    compute_metrics,
    compute_rd_metrics,
    MetricsTracker,
)
from .plotting import (
    plot_rd_curve,
    plot_reconstructions,
    plot_training_curves,
    plot_latent_space_2d,
    setup_plotting_style,
)
from .seed import set_seed, get_random_state, set_random_state

__all__ = [
    # Dataset utilities
    "get_bin_mnist_loaders",
    "visualize_batch",
    "compute_dataset_stats",
    # Metrics
    "compute_rate",
    "compute_distortion",
    "compute_elbo",
    "compute_metrics",
    "compute_rd_metrics",
    "MetricsTracker",
    # Plotting
    "plot_rd_curve",
    "plot_reconstructions",
    "plot_training_curves",
    "plot_latent_space_2d",
    "setup_plotting_style",
    # Seed utilities
    "set_seed",
    "get_random_state",
    "set_random_state",
]
