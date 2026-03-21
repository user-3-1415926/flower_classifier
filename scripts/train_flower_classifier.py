import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flower_classifier.config import (  # noqa: E402
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CONTINUE_EPOCHS,
    CONTINUE_LR,
    CONTINUE_MODEL_PATH,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    HISTORY_PATH,
    LABEL_SMOOTHING,
    MODEL_DIR,
    NUM_WORKERS,
    PRETRAINED_MODEL_PATH,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_EPOCHS,
    STAGE2_LR,
    TEST_DIR,
    TRAIN_DIR,
    USE_TTA,
    WEIGHT_DECAY,
)
from flower_classifier.model import FlowerVGG19, build_transforms, predict_with_tta  # noqa: E402
from flower_classifier.utils import (  # noqa: E402
    calculate_accuracy,
    count_correct_predictions,
    get_device,
    get_result_dir,
    save_history_csv,
    save_history_json,
    save_summary_json,
    save_training_curves,
    set_seed,
)


SEED = 42
EXPERIMENT_NAME = "flower_vgg19"
CHECKPOINT_EVERY = 10


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_score = None
        self.bad_epochs = 0

    def step(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class TrainingManager:
    def __init__(self, model, history, best_test_acc, device, result_dir):
        self.model = model
        self.history = history
        self.best_test_acc = best_test_acc
        self.device = device
        self.result_dir = result_dir

    def run_phase(self, train_loader, test_loader, criterion, phase_config, start_epoch):
        optimizer = build_optimizer(self.model.get_trainable_parameters(), phase_config["lr"])
        scheduler = build_scheduler(optimizer)
        early_stopper = EarlyStopping(EARLY_STOPPING_PATIENCE)
        end_epoch = start_epoch + phase_config["epochs"]

        for epoch in range(start_epoch + 1, end_epoch + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch}/{end_epoch} | stage={phase_config['name']} | lr={current_lr:.6g}")
            epoch_start_time = time.perf_counter()

            avg_train_loss, avg_train_acc = run_epoch(
                self.model,
                train_loader,
                criterion,
                self.device,
                optimizer=optimizer,
                epoch=epoch,
                stage_name=phase_config["name"],
                use_tta=False,
            )
            avg_test_loss, avg_test_acc = run_epoch(
                self.model,
                test_loader,
                criterion,
                self.device,
                epoch=epoch,
                stage_name=phase_config["name"],
                use_tta=USE_TTA,
            )
            epoch_duration = time.perf_counter() - epoch_start_time

            improved = self._maybe_save_best(avg_test_acc)
            self._record_epoch(
                train_loss=avg_train_loss,
                train_acc=avg_train_acc,
                test_loss=avg_test_loss,
                test_acc=avg_test_acc,
                learning_rate=current_lr,
            )
            scheduler.step(avg_test_acc)

            print(
                f"Epoch {epoch}/{end_epoch} [{phase_config['name']}]: "
                f"Train Acc: {avg_train_acc:.3f}, Loss: {avg_train_loss:.4f} | "
                f"Test Acc: {avg_test_acc:.3f}, Loss: {avg_test_loss:.4f} | "
                f"Best: {self.best_test_acc:.3f} | "
                f"LR: {current_lr:.6g} | "
                f"Time: {format_duration(epoch_duration)}"
            )

            self._maybe_save_checkpoint(epoch, phase_config["checkpoint_prefix"], end_epoch)
            self._save_artifacts(epoch, avg_train_loss, avg_test_loss, phase_config["name"], current_lr)

            if early_stopper.step(avg_test_acc):
                print(
                    f"Early stopping triggered in {phase_config['name']} after "
                    f"{EARLY_STOPPING_PATIENCE} epochs without validation improvement."
                )
                return epoch, improved

        return end_epoch, False

    def _record_epoch(self, train_loss, train_acc, test_loss, test_acc, learning_rate):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["test_loss"].append(test_loss)
        self.history["test_acc"].append(test_acc)
        self.history["best_test_acc"].append(self.best_test_acc)
        self.history["learning_rate"].append(learning_rate)

    def _maybe_save_best(self, avg_test_acc):
        if avg_test_acc <= self.best_test_acc:
            return False

        self.best_test_acc = avg_test_acc
        torch.save(self.model.state_dict(), BEST_MODEL_PATH)
        torch.save(self.model.state_dict(), CONTINUE_MODEL_PATH)
        print(f"saved improved best model: {BEST_MODEL_PATH}")
        return True

    def _maybe_save_checkpoint(self, epoch, checkpoint_prefix, final_epoch):
        if epoch % CHECKPOINT_EVERY != 0 and epoch != final_epoch:
            return

        checkpoint_path = os.path.join(MODEL_DIR, f"{checkpoint_prefix}_epoch{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"saved checkpoint: {checkpoint_path}")

    def _save_artifacts(self, epoch, train_loss, test_loss, stage_name, learning_rate):
        save_history_json(self.history, HISTORY_PATH)
        save_history_csv(self.history, self.result_dir / "history.csv")
        save_training_curves(self.history, self.result_dir / "training_curves.png")
        save_summary_json(
            {
                "epoch": epoch,
                "best_test_acc": self.best_test_acc,
                "latest_train_loss": train_loss,
                "latest_test_loss": test_loss,
                "stage": stage_name,
                "seed": SEED,
                "device": str(self.device),
                "learning_rate": learning_rate,
                "tta_enabled": USE_TTA,
            },
            self.result_dir / "summary.json",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train or continue training the flower VGG19 classifier.")
    parser.add_argument(
        "--resume-best",
        action="store_true",
        help="Continue training from the current best checkpoint using the continue phase settings.",
    )
    return parser.parse_args()


def resolve_num_workers():
    if os.name == "nt":
        return 0
    return NUM_WORKERS


def create_empty_history():
    return {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "best_test_acc": [],
        "learning_rate": [],
    }


def load_history(history_path):
    if not os.path.exists(history_path):
        return create_empty_history()

    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)

    defaults = create_empty_history()
    epoch_count = len(history.get("train_loss", []))

    for key, default_values in defaults.items():
        history.setdefault(key, list(default_values))

    if len(history["best_test_acc"]) < epoch_count:
        running_best = 0.0
        for test_acc in history.get("test_acc", []):
            running_best = max(running_best, test_acc)
            history["best_test_acc"].append(running_best)

    if len(history["learning_rate"]) < epoch_count:
        history["learning_rate"].extend([""] * (epoch_count - len(history["learning_rate"])))

    return history


def build_optimizer(parameters, learning_rate):
    return optim.AdamW(parameters, lr=learning_rate, weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer):
    return ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )


def run_epoch(model, data_loader, criterion, device, optimizer=None, epoch=None, stage_name=None, use_tta=False):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    mode_name = "train" if is_training else "val"

    total_loss = 0.0
    total_correct = 0
    total_samples = len(data_loader.dataset)

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        progress_bar = tqdm(
            enumerate(data_loader, start=1),
            total=len(data_loader),
            desc=f"Epoch {epoch} {stage_name} {mode_name}",
            leave=False,
        )
        for batch_idx, (data, label) in progress_bar:
            data, label = data.to(device), label.to(device)

            if is_training:
                optimizer.zero_grad()
                output = model(data)
            else:
                output = predict_with_tta(model, data) if use_tta else model(data)

            loss = criterion(output, label)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_correct += count_correct_predictions(output, label)
            avg_so_far = total_loss / batch_idx
            acc_so_far = calculate_accuracy(total_correct, batch_idx * data.size(0))
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_so_far:.4f}",
                acc=f"{acc_so_far:.3f}",
            )

    avg_loss = total_loss / len(data_loader)
    avg_acc = calculate_accuracy(total_correct, total_samples)
    return avg_loss, avg_acc


def format_duration(seconds):
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def build_data_loaders():
    train_transform, test_transform = build_transforms()
    num_workers = resolve_num_workers()

    train_dataset = datasets.ImageFolder(TRAIN_DIR, train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataset, test_dataset, train_loader, test_loader, num_workers


def create_model(class_num, device):
    model = FlowerVGG19(
        class_num,
        model_path=PRETRAINED_MODEL_PATH,
        dropout=DROPOUT,
    ).to(device)
    return model


def print_run_header(device, train_dataset, test_dataset, class_num, num_workers, resume_best):
    print("Continuing flower classifier training" if resume_best else "Starting flower classifier training")
    print(f"device = {device}")
    print(f"train_data size = {len(train_dataset)}")
    print(f"test_data size = {len(test_dataset)}")
    print(f"class_num = {class_num}")
    print(f"batch_size = {BATCH_SIZE}")
    print(f"num_workers = {num_workers}")
    print(f"label_smoothing = {LABEL_SMOOTHING}")
    print(f"weight_decay = {WEIGHT_DECAY}")
    print(f"tta = {USE_TTA}")


def run_fresh_training(manager, train_loader, test_loader, criterion):
    phase1 = {
        "name": "stage1_classifier",
        "epochs": STAGE1_EPOCHS,
        "lr": STAGE1_LR,
        "checkpoint_prefix": "flowers",
    }
    phase2 = {
        "name": "stage2_finetune",
        "epochs": STAGE2_EPOCHS,
        "lr": STAGE2_LR,
        "checkpoint_prefix": "flowers",
    }

    # Stage 1 keeps the ImageNet backbone frozen so the new classifier can stabilize first.
    manager.model.freeze_features()
    last_epoch, _ = manager.run_phase(train_loader, test_loader, criterion, phase1, start_epoch=0)

    # Stage 2 opens the last two VGG blocks to gain a bit more task-specific capacity.
    manager.model.unfreeze_last_two_blocks()
    print("Switched to stage 2: unfroze the last two convolutional blocks")
    manager.run_phase(train_loader, test_loader, criterion, phase2, start_epoch=last_epoch)


def run_continue_training(manager, train_loader, test_loader, criterion):
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model checkpoint not found: {BEST_MODEL_PATH}")

    manager.model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=manager.device))
    manager.model.unfreeze_last_two_blocks()

    start_epoch = len(manager.history["train_loss"])
    target_total_epochs = start_epoch + CONTINUE_EPOCHS
    print(f"resume_model = {BEST_MODEL_PATH}")
    print(f"start_epoch = {start_epoch}")
    print(f"continue_epochs = {CONTINUE_EPOCHS}")
    print(f"target_total_epochs = {target_total_epochs}")
    print(f"continue_lr = {CONTINUE_LR}")

    phase = {
        "name": "continue_finetune",
        "epochs": CONTINUE_EPOCHS,
        "lr": CONTINUE_LR,
        "checkpoint_prefix": "flowers_continue",
    }
    manager.run_phase(train_loader, test_loader, criterion, phase, start_epoch=start_epoch)


def main(resume_best=False):
    set_seed(SEED)
    train_dataset, test_dataset, train_loader, test_loader, num_workers = build_data_loaders()

    class_num = len(train_dataset.classes)
    device = get_device()
    history = load_history(HISTORY_PATH) if resume_best else create_empty_history()
    best_test_acc = max(history["best_test_acc"], default=0.0)

    print_run_header(device, train_dataset, test_dataset, class_num, num_workers, resume_best)

    model = create_model(class_num, device)
    os.makedirs(MODEL_DIR, exist_ok=True)
    result_dir = get_result_dir(MODEL_DIR, EXPERIMENT_NAME)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    manager = TrainingManager(model, history, best_test_acc, device, result_dir)

    if resume_best:
        run_continue_training(manager, train_loader, test_loader, criterion)
    else:
        print(
            f"stage1 = {STAGE1_EPOCHS} epochs @ lr={STAGE1_LR} | "
            f"stage2 = {STAGE2_EPOCHS} epochs @ lr={STAGE2_LR} | "
            f"continue = {CONTINUE_EPOCHS} epochs @ lr={CONTINUE_LR}"
        )
        run_fresh_training(manager, train_loader, test_loader, criterion)

    print(f"\nTraining finished. Best test accuracy = {manager.best_test_acc:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(resume_best=args.resume_best)
