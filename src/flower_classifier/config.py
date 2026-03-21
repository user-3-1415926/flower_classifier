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
CONTINUE_MODEL_PATH = os.path.join(MODEL_DIR, "flowers_continue_best.pth")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")

ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
SAMPLE_IMAGE_PATH = os.path.join(ASSETS_DIR, "sample_image.jpg")
IMAGENET_LABELS_PATH = os.path.join(ASSETS_DIR, "imagenet_classes.txt")
COCO_CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "metadata", "captions_train2017.json")

BATCH_SIZE = 32
NUM_WORKERS = 4

# Keep the original two-stage framework, but make the second stage a bit stronger.
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 15
CONTINUE_EPOCHS = 10

STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
CONTINUE_LR = 1e-5

DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR = 1e-6

EARLY_STOPPING_PATIENCE = 5
USE_TTA = True

# Backward-compatible aliases for existing imports.
CLASSIFIER_DROPOUT = DROPOUT
EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS
LEARNING_RATE = STAGE1_LR
