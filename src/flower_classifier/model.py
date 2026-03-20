import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import VGG19_Weights


class FlowerVGG19(nn.Module):
    def __init__(self, num_classes, model_path=None, dropout=0.5, use_pretrained=False):
        super().__init__()
        self.model = self._build_backbone(model_path, use_pretrained)
        self._replace_classifier(num_classes, dropout)

    def _build_backbone(self, model_path=None, use_pretrained=False):
        if model_path:
            model = models.vgg19(weights=None)
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            return model

        if use_pretrained:
            return models.vgg19(weights=VGG19_Weights.DEFAULT)

        return models.vgg19(weights=None)

    def _replace_classifier(self, num_classes, dropout):
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def freeze_features(self):
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze_last_conv_block(self):
        self.freeze_features()
        for layer in self.model.features[28:]:
            for param in layer.parameters():
                param.requires_grad = True

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def forward(self, x):
        return self.model(x)


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_transform = transforms.Compose(
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

    return train_transform, test_transform
