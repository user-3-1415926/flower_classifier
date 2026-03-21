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

    def unfreeze_last_block(self):
        self._unfreeze_last_n_blocks(1)

    def unfreeze_last_two_blocks(self):
        self._unfreeze_last_n_blocks(2)

    def _unfreeze_last_n_blocks(self, num_blocks):
        self.freeze_features()
        block_ranges = self._get_feature_block_ranges()
        for start_idx, end_idx in block_ranges[-num_blocks:]:
            for layer in self.model.features[start_idx:end_idx]:
                for param in layer.parameters():
                    param.requires_grad = True

    def _get_feature_block_ranges(self):
        block_ranges = []
        block_start = 0

        for layer_idx, layer in enumerate(self.model.features):
            if isinstance(layer, nn.MaxPool2d):
                block_ranges.append((block_start, layer_idx + 1))
                block_start = layer_idx + 1

        if block_start < len(self.model.features):
            block_ranges.append((block_start, len(self.model.features)))

        return block_ranges

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def forward(self, x):
        return self.model(x)


def predict_with_tta(model, inputs):
    # A minimal and stable TTA: average logits from the original image and its mirror.
    logits = model(inputs)
    flipped_logits = model(torch.flip(inputs, dims=[3]))
    return (logits + flipped_logits) / 2.0


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
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
