import json
import os
import random
import sys
from pathlib import Path

import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flower_classifier.config import COCO_CAPTIONS_PATH, TRAIN_DIR  # noqa: E402


def download_coco_none_samples(json_path, output_dir, num_samples=1500):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    sample_size = min(num_samples, len(images))
    selected_images = random.sample(images, sample_size)
    print(f"Selected {sample_size} images. Starting download...")

    count = 0
    for img_info in tqdm(selected_images):
        file_name = img_info["file_name"]
        url = img_info["coco_url"]
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(os.path.join(output_dir, file_name), "wb") as f:
                    f.write(response.content)
                count += 1
        except Exception:
            continue

    print(f"Download complete. Saved {count} images to {output_dir}.")


if __name__ == "__main__":
    download_coco_none_samples(
        json_path=COCO_CAPTIONS_PATH,
        output_dir=os.path.join(TRAIN_DIR, "none"),
        num_samples=500,
    )
