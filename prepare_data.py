from torchvision.datasets import CIFAR10
from torchvision import transforms

if __name__ == "__main__":

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "CIFAR10",
        train=True,
        download=True,
        transform=train_transforms,
    )