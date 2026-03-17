import os
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flower_classifier.config import BEST_MODEL_PATH, TEST_DIR  # noqa: E402
from flower_classifier.model import FlowerVGG19  # noqa: E402


FALLBACK_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "flowers_epoch40.pth")


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_data = datasets.ImageFolder(TEST_DIR, transform)
    print(f"test_data size = {len(test_data)}")

    class_num = len(test_data.classes)
    label2index = test_data.class_to_idx
    index2label = {idx: label for label, idx in label2index.items()}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else FALLBACK_MODEL_PATH
    vgg19 = FlowerVGG19(class_num).to(device)
    vgg19.load_state_dict(torch.load(model_path, map_location=device))
    vgg19.eval()
    print(f"loaded model = {model_path}")

    correct = 0
    total = 0

    for i, (data, label) in enumerate(test_data):
        data = data.unsqueeze(0).to(device)
        label = torch.tensor([label], dtype=torch.long).to(device)

        predict = vgg19(data).argmax(1)

        if predict.eq(label).item():
            correct += 1
        else:
            predict_str = index2label[predict.item()]
            file_path, _ = test_data.imgs[i]
            print(f"wrong case: {file_path}\t{predict_str}")

        total += 1

    acc = 100 * correct / total
    print(f"Accuracy: {correct}/{total}={acc:.3f}%")
