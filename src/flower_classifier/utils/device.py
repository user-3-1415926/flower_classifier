"""Device selection helpers."""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the preferred computation device for the current machine."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
