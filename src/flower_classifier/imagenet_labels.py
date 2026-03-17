def load_imagenet_labels(path):
    id_to_label = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            key, value = line.strip().split(",")
            id_to_label[int(key)] = value
    return id_to_label


if __name__ == "__main__":
    imagenet_id_to_labels = load_imagenet_labels("imagenet_classes.txt")
    print(imagenet_id_to_labels)
