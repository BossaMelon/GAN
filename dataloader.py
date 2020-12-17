from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CelebA

from utils.path_handle import data_path


def get_dataloader_mnist(batch_size, transform=transforms.ToTensor()):
    dataloader = DataLoader(
        MNIST(root=str(data_path), download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True)
    return dataloader


def get_dataloader_celebA(batch_size, transform):
    dataloader = DataLoader(
        CelebA(root=str(data_path), split='train', download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True)
    return dataloader


if __name__ == '__main__':
    pass
