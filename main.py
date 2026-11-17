...

from datasets import load_dataset
from torch.utils.data import DataLoader

from dataset import CIFAR10Dataset, load_data
from dataset import train_transform, test_transform
from model import VIT

def main():
    # DATA

    train_images, train_labels, test_images, test_labels = load_data()

    train_dataset = CIFAR10Dataset(train_images, train_labels, transform=train_transform)
    test_dataset = CIFAR10Dataset(test_images, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=4)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=128, num_workers=4)

    C, H, W = train_loader.dataset[0][0].shape

    
    # MODEL INIT
    
    model = VIT(H=H, C=C, P=4, D=192)

    images, labels = next(iter(train_loader))

    out = model(images)

    print(out.shape)


if __name__=="__main__":
    main()