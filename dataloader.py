from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from util import data_path


def get_dataloader(batch_size, transform=transforms.ToTensor()):
    dataloader = DataLoader(
        MNIST(root=str(data_path), download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True)
    return dataloader


if __name__ == '__main__':
    pass
