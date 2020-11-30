import torch
import torch.nn as nn

from dataloader import get_dataloader
from models import Generator, Discriminator
from train import train

from experiments import gan

if __name__ == '__main__':
    gan()