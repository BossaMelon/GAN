import torch
import torch.nn as nn

from dataloader import get_dataloader
from models import Generator, Discriminator
from train import train


def gan():
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    z_dim = 64
    batch_size = 128
    lr = 0.00001

    gen = Generator(z_dim)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator()
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    dataloader = get_dataloader(batch_size)

    train(gen, disc, dataloader, n_epochs, gen_opt, disc_opt, criterion, z_dim)


def dcgan():
    pass


if __name__ == '__main__':
    gan()
