import torch
from torchvision import datasets, transforms

# Create a common folder for all datasets
root = "./dl_dataset"

# Convert images into tensors
transform = transforms.ToTensor()


# Download MNIST dataset
mnist_train = datasets.MNIST(
    root=root,
    train=True,
    download=True,
    transform=transform
)

mnist_test = datasets.MNIST(
    root=root,
    train=False,
    download=True,
    transform=transform
)


# Download Fashion-MNIST dataset
fashion_train = datasets.FashionMNIST(
    root=root,
    train=True,
    download=True,
    transform=transform
)

fashion_test = datasets.FashionMNIST(
    root=root,
    train=False,
    download=True,
    transform=transform
)


# Download CIFAR-10 dataset
cifar_train = datasets.CIFAR10(
    root=root,
    train=True,
    download=True,
    transform=transform
)

cifar_test = datasets.CIFAR10(
    root=root,
    train=False,
    download=True,
    transform=transform
)


# Print dataset sizes
print("MNIST training images:", len(mnist_train))
print("MNIST testing images:", len(mnist_test))

print("Fashion-MNIST training images:", len(fashion_train))
print("Fashion-MNIST testing images:", len(fashion_test))

print("CIFAR-10 training images:", len(cifar_train))
print("CIFAR-10 testing images:", len(cifar_test))

