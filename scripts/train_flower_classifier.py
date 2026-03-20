import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flower_classifier.config import (  # noqa: E402
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASSIFIER_DROPOUT,
    HISTORY_PATH,
    MODEL_DIR,
    NUM_WORKERS,
    PRETRAINED_MODEL_PATH,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_EPOCHS,
    STAGE2_LR,
    TEST_DIR,
    TRAIN_DIR,
)
from flower_classifier.model import FlowerVGG19, build_transforms  # noqa: E402
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


def run_epoch(model, data_loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = len(data_loader.dataset)

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)

            if is_training:
                optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, label)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_correct += count_correct_predictions(output, label)

    avg_loss = total_loss / len(data_loader)
    avg_acc = calculate_accuracy(total_correct, total_samples)
    return avg_loss, avg_acc


if __name__ == "__main__":
    set_seed(SEED)
    train_transform, test_transform = build_transforms()

    train_dataset = datasets.ImageFolder(TRAIN_DIR, train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, test_transform)

    print(f"train_data size = {len(train_dataset)}")
    print(f"test_data size = {len(test_dataset)}")

    class_num = len(train_dataset.classes)
    print(f"class_num = {class_num}")

    train_load = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_load = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = get_device()
    print(f"device = {device}")

    vgg19 = FlowerVGG19(
        class_num,
        model_path=PRETRAINED_MODEL_PATH,
        dropout=CLASSIFIER_DROPOUT,
    ).to(device)
    vgg19.freeze_features()

    stage1_optimizer = optim.Adam(
        vgg19.get_trainable_parameters(),
        lr=STAGE1_LR,
    )
    stage2_optimizer = optim.Adam(
        vgg19.get_trainable_parameters(),
        lr=STAGE2_LR,
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(MODEL_DIR, exist_ok=True)
    result_dir = get_result_dir(MODEL_DIR, EXPERIMENT_NAME)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_test_acc = 0.0
    total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS

    for epoch in range(1, total_epochs + 1):
        if epoch == STAGE1_EPOCHS + 1:
            vgg19.unfreeze_last_conv_block()
            stage2_optimizer = optim.Adam(
                vgg19.get_trainable_parameters(),
                lr=STAGE2_LR,
            )

        optimizer = stage1_optimizer if epoch <= STAGE1_EPOCHS else stage2_optimizer
        stage_name = "stage1_classifier" if epoch <= STAGE1_EPOCHS else "stage2_finetune"

        avg_train_loss, avg_train_acc = run_epoch(
            vgg19,
            train_load,
            criterion,
            device,
            optimizer=optimizer,
        )
        avg_test_loss, avg_test_acc = run_epoch(
            vgg19,
            test_load,
            criterion,
            device,
        )

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["test_loss"].append(avg_test_loss)
        history["test_acc"].append(avg_test_acc)

        print(
            f"Epoch {epoch}/{total_epochs} [{stage_name}]: "
            f"Train Acc: {avg_train_acc:.3f}, Loss: {avg_train_loss:.4f} | "
            f"Test Acc: {avg_test_acc:.3f}, Loss: {avg_test_loss:.4f}"
        )

        if avg_test_acc > best_test_acc:
            best_test_acc = avg_test_acc
            torch.save(vgg19.state_dict(), BEST_MODEL_PATH)
            print(f"saved best model: {BEST_MODEL_PATH}")

        if epoch % 10 == 0:
            model_name = os.path.join(MODEL_DIR, f"flowers_epoch{epoch}.pth")
            torch.save(vgg19.state_dict(), model_name)
            print(f"saved checkpoint: {model_name}")

        save_history_json(history, HISTORY_PATH)
        save_history_csv(history, result_dir / "history.csv")
        save_training_curves(history, result_dir / "training_curves.png")
        save_summary_json(
            {
                "epoch": epoch,
                "best_test_acc": best_test_acc,
                "latest_train_loss": avg_train_loss,
                "latest_test_loss": avg_test_loss,
                "stage": stage_name,
                "seed": SEED,
                "device": str(device),
            },
            result_dir / "summary.json",
        )
