"""Plot helpers for experiment history."""

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


HistoryDict = Mapping[str, Sequence[float]]


def plot_training_curves(history: HistoryDict, save_path: str | Path) -> None:
    """Save loss and accuracy curves from a training history dictionary."""
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.get("train_loss", []), label="train_loss")
    plt.plot(epochs, history.get("test_loss", []), label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.get("train_acc", []), label="train_acc")
    plt.plot(epochs, history.get("test_acc", []), label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
