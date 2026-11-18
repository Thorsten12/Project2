from datasets import load_dataset
from torch.utils.data import DataLoader

import argparse

from dataset import CIFAR10Dataset, load_data
from dataset import train_transform, test_transform
from model import FullModel

def train():
    # PARSER

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--hidden_dim", default=192, type=int)
    parser.add_argument("--num_heads", default=3, type=int)
    parser.add_argument("--mlp_mult", default=2, type=float)
    parser.add_argument("--classifier_dim", default=192, type=int)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--optimizer")
    parser.add_argument("--lr")
    parser.add_argument("--warmup_epoch")

    args = parser.parse_args()

    # DATA

    train_images, train_labels, test_images, test_labels = load_data()

    train_dataset = CIFAR10Dataset(train_images, train_labels, transform=train_transform)
    test_dataset = CIFAR10Dataset(test_images, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    C, H, W = train_loader.dataset[0][0].shape

    
    # MODEL INIT
    
    model = FullModel(H=H, C=C, P=args.patch_size, D=args.hidden_dim, num_heads=args.num_heads, mlp_mult=args.mlp_mult, classifier_dim=args.classifier_dim)

    images, labels = next(iter(train_loader))

    out = model(images)

    print(out.shape)


if __name__=="__main__":
    train()