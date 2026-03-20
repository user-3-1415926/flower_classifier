import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "archive", "flowers")
TRAIN_DIR = os.path.join(ARCHIVE_DIR, "train")
TEST_DIR = os.path.join(ARCHIVE_DIR, "test")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PRETRAINED_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "pretrained",
    "hub",
    "checkpoints",
    "vgg19-dcbb9e9d.pth",
)
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "flowers_best.pth")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")

ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
SAMPLE_IMAGE_PATH = os.path.join(ASSETS_DIR, "sample_image.jpg")
IMAGENET_LABELS_PATH = os.path.join(ASSETS_DIR, "imagenet_classes.txt")
COCO_CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "metadata", "captions_train2017.json")

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_WORKERS = 4
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 10
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
CLASSIFIER_DROPOUT = 0.5
