# Flower Classifier Project

This project trains a flower image classifier with a VGG19 backbone using PyTorch.

## Project Structure

```text
clfar_10/
├─ archive/
│  └─ flowers/
│     ├─ train/
│     └─ test/
├─ assets/
│  ├─ imagenet_classes.txt
│  └─ sample_image.jpg
├─ metadata/
│  └─ captions_train2017.json
├─ models/
├─ pretrained/
├─ scripts/
│  ├─ train_flower_classifier.py
│  ├─ evaluate_flower_classifier.py
│  ├─ split_flower_dataset.py
│  ├─ download_none_images.py
│  ├─ download_vgg19_weights.py
│  └─ predict_with_imagenet.py
└─ src/
   └─ flower_classifier/
      ├─ config.py
      ├─ model.py
      └─ imagenet_labels.py
```

## Main Workflows

### 1. Download pretrained VGG19 weights

```powershell
D:\Anaconda\python.exe scripts\download_vgg19_weights.py
```

### 2. Split dataset into train/test

Use this only when `archive/flowers/` still contains unsplit class folders.

```powershell
D:\Anaconda\python.exe scripts\split_flower_dataset.py
```

### 3. Download extra `none` class images

This downloads random COCO images into `archive/flowers/train/none`.

```powershell
D:\Anaconda\python.exe scripts\download_none_images.py
```

### 4. Train the flower classifier

```powershell
D:\Anaconda\python.exe scripts\train_flower_classifier.py
```

Training outputs:

- `models/flowers_best.pth`
- `models/flowers_epoch10.pth`, `flowers_epoch20.pth`, ...
- `models/history.json`

### 5. Evaluate the trained model

```powershell
D:\Anaconda\python.exe scripts\evaluate_flower_classifier.py
```

### 6. Run single-image ImageNet prediction

This is only for checking the original pretrained VGG19 model, not your flower classifier.

```powershell
D:\Anaconda\python.exe scripts\predict_with_imagenet.py
```

## Key Files

- `src/flower_classifier/config.py`: paths and training hyperparameters
- `src/flower_classifier/model.py`: VGG19 model definition and transforms
- `src/flower_classifier/imagenet_labels.py`: ImageNet label loading helper

## Notes

- The training dataset is expected at `archive/flowers/train`.
- The test dataset is expected at `archive/flowers/test`.
- The project currently uses the manually defined VGG19 architecture and loads pretrained weights from `pretrained/hub/checkpoints/`.
