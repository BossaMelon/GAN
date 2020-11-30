import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

data_root = './data'


def get_dataloader(batch_size):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    dataloader = DataLoader(
        MNIST(root=data_root, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True)
    return dataloader


if __name__ == '__main__':
    get_dataloader(64)
