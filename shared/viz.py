"""Plotting helpers for HTB AI Red Team exercises."""

from __future__ import annotations


def plot_loss_curves(train_losses, test_losses=None, title: str = "Loss"):
    """Plot training (and optional test) loss per epoch."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    if test_losses is not None:
        ax.plot(test_losses, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig, ax
