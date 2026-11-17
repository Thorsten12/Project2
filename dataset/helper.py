from torchvision import transforms as T

### AUGMENTATION

train_transform = T.Compose([
    T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
    T.ToTensor(),
    T.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])
