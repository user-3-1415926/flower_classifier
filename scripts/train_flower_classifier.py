import json
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
    EPOCHS,
    HISTORY_PATH,
    LEARNING_RATE,
    MODEL_DIR,
    NUM_WORKERS,
    PRETRAINED_MODEL_PATH,
    TEST_DIR,
    TRAIN_DIR,
)
from flower_classifier.model import FlowerVGG19, build_transforms  # noqa: E402


if __name__ == "__main__":
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    vgg19 = FlowerVGG19(class_num, PRETRAINED_MODEL_PATH).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, vgg19.parameters()),
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(MODEL_DIR, exist_ok=True)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_test_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        vgg19.train()
        train_correct = 0
        train_loss = 0.0

        for batch_idx, (data, label) in enumerate(train_load):
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            output = vgg19(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            predict = torch.argmax(output, dim=1)
            train_correct += (predict == label).sum().item()
            train_loss += loss.item()
            print(f"{batch_idx + 1}/{len(train_load)}")

        avg_train_loss = train_loss / len(train_load)
        avg_train_acc = train_correct / len(train_dataset)

        vgg19.eval()
        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for data, label in test_load:
                data, label = data.to(device), label.to(device)
                output = vgg19(data)
                loss = criterion(output, label)

                test_loss += loss.item()
                predict = torch.argmax(output, dim=1)
                test_correct += (predict == label).sum().item()

        avg_test_loss = test_loss / len(test_load)
        avg_test_acc = test_correct / len(test_dataset)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["test_loss"].append(avg_test_loss)
        history["test_acc"].append(avg_test_acc)

        print(
            f"Epoch {epoch}/{EPOCHS}: "
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

        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
