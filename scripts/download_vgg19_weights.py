import os

from torchvision import models
from torchvision.models import VGG19_Weights


PRETRAINED_DIR = "./pretrained"


os.environ["TORCH_HOME"] = PRETRAINED_DIR
os.makedirs(PRETRAINED_DIR, exist_ok=True)

model = models.vgg19(weights=VGG19_Weights.DEFAULT).eval()


if __name__ == "__main__":
    print(model)
