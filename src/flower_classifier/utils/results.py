"""Helpers for saving training artifacts and summaries."""

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .plot import plot_training_curves


HistoryDict = Mapping[str, Sequence[float]]


def get_result_dir(base_dir: str | Path, experiment_name: str = "flower_vgg19") -> Path:
    """Return the directory used to store one experiment's artifacts."""
    result_dir = Path(base_dir) / experiment_name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def save_history_json(history: HistoryDict, save_path: str | Path) -> None:
    """Save history data to a JSON file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def save_history_csv(history: HistoryDict, save_path: str | Path) -> None:
    """Save history data to a CSV file."""
    rows = _history_rows(history)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "test_loss",
                "train_acc",
                "test_acc",
                "best_test_acc",
                "learning_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_training_curves(history: HistoryDict, save_path: str | Path) -> None:
    """Save the training curves figure."""
    plot_training_curves(history, save_path)


def save_summary_json(summary: Mapping[str, Any], save_path: str | Path) -> None:
    """Save a summary dictionary to a JSON file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def _history_rows(history: HistoryDict) -> list[dict[str, float | int]]:
    epochs = len(history.get("train_loss", []))
    rows: list[dict[str, float | int]] = []

    for index in range(epochs):
        rows.append(
            {
                "epoch": index + 1,
                "train_loss": history["train_loss"][index],
                "test_loss": history["test_loss"][index],
                "train_acc": history["train_acc"][index],
                "test_acc": history["test_acc"][index],
                "best_test_acc": _history_value(history, "best_test_acc", index, history["test_acc"][index]),
                "learning_rate": _history_value(history, "learning_rate", index, ""),
            }
        )

    return rows


def _history_value(history: HistoryDict, key: str, index: int, default: float | str) -> float | str:
    values = history.get(key, [])
    if index < len(values):
        return values[index]
    return default
