import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from util import project_root


data_root = project_root/'data'


def get_dataloader(batch_size, transform=transforms.ToTensor()):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    dataloader = DataLoader(
        MNIST(root=str(data_root), download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True)
    return dataloader


if __name__ == '__main__':
    get_dataloader(64)
