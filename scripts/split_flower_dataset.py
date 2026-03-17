import random
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flower_classifier.config import ARCHIVE_DIR, TEST_DIR, TRAIN_DIR  


TEST_RATIO = 0.2
SEED = 42


def is_leaf_class_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name in {"train", "test"}:
        return False
    return any(child.is_file() for child in path.iterdir())


def split_class_dir(class_dir: Path) -> None:
    images = [item for item in class_dir.iterdir() if item.is_file()]
    if not images:
        return

    random.shuffle(images)
    test_count = max(1, int(len(images) * TEST_RATIO))
    train_images = images[test_count:]
    test_images = images[:test_count]

    train_class_dir = Path(TRAIN_DIR) / class_dir.name
    test_class_dir = Path(TEST_DIR) / class_dir.name
    train_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    for image_path in train_images:
        shutil.move(str(image_path), str(train_class_dir / image_path.name))

    for image_path in test_images:
        shutil.move(str(image_path), str(test_class_dir / image_path.name))

    class_dir.rmdir()


def main() -> None:
    random.seed(SEED)

    source_dir = Path(ARCHIVE_DIR)
    class_dirs = [path for path in source_dir.iterdir() if is_leaf_class_dir(path)]
    if not class_dirs:
        print("No unsplit class folders found under archive/flowers.")
        return

    Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEST_DIR).mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(class_dirs):
        split_class_dir(class_dir)
        print(f"split {class_dir.name}")

    print("Dataset split complete.")


if __name__ == "__main__":
    main()
