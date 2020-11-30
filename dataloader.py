from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_dataloader(batch_size):
    dataloader = DataLoader(
        MNIST(root='./data', download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True)
    return dataloader


if __name__ == '__main__':
    get_dataloader(64)
