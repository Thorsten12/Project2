import torch
from torch.utils.data import Dataset
from PIL import Image

from datasets import load_dataset

def load_data():

    dataset = load_dataset("uoft-cs/cifar10")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print("Size of TrainDataset: ", len(train_dataset))  # 50000
    print("Size of TrainDataset: ", len(test_dataset))   # 10000


    import numpy as np

    hf_train = dataset["train"]
    hf_test = dataset["test"]

    train_images = np.stack([
        np.array(img.convert("RGB"))
        for img in hf_train["img"]
    ]).astype(np.float32) / 255.0

    train_labels = np.array(hf_train["label"])

    test_images = np.stack([
        np.array(img.convert("RGB"))
        for img in hf_test["img"]
    ]).astype(np.float32) / 255.0

    test_labels = np.array(hf_test["label"])

    return train_images, train_labels, test_images, test_labels

class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray((image * 255).astype("uint8"))

        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
