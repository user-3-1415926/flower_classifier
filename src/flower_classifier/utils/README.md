# Utils Module

This directory contains small reusable helpers for the flower classifier project.

## Modules

- `device.py`: choose the runtime device, such as CPU or CUDA
- `metrics.py`: compute correct predictions and accuracy
- `plot.py`: generate and save training curve figures
- `results.py`: save history, summaries, and other experiment artifacts
- `seed.py`: set random seeds for reproducible runs
- `__init__.py`: re-export the main helpers for convenient imports

## Typical Usage

```python
from flower_classifier.utils import (
    get_device,
    set_seed,
    calculate_accuracy,
    count_correct_predictions,
    save_history_json,
)
```

These helpers are used by the training, evaluation, and dataset scripts to keep shared logic modular and easy to maintain.
