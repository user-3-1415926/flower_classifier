import sys
from pathlib import Path

from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flower_classifier.config import IMAGENET_LABELS_PATH, SAMPLE_IMAGE_PATH  # noqa: E402
from flower_classifier.imagenet_labels import load_imagenet_labels  # noqa: E402
from flower_classifier.model import build_transforms  # noqa: E402

from download_vgg19_weights import model  # noqa: E402


if __name__ == "__main__":
    image = Image.open(SAMPLE_IMAGE_PATH)

    _, transform = build_transforms()

    image_tensor = transform(image).unsqueeze(0)
    output = model(image_tensor)
    _, predict = torch.max(output, 1)

    imagenet_id_to_label = load_imagenet_labels(IMAGENET_LABELS_PATH)

    print(output.shape)
    print(predict)
    print(imagenet_id_to_label[predict.item()])
