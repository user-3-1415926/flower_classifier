"""Metric helpers used across training and evaluation."""

import torch


def count_correct_predictions(predictions: torch.Tensor, labels: torch.Tensor) -> int:
    """Count how many predicted labels match the targets."""
    predicted_labels = predictions.argmax(dim=1)
    return int((predicted_labels == labels).sum().item())


def calculate_accuracy(correct: int, total: int) -> float:
    """Compute accuracy as a ratio in the range [0, 1]."""
    if total <= 0:
        return 0.0
    return correct / total
