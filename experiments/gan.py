import torch
import torch.nn as nn

from dataloader import get_dataloader
from models.model_gan import Generator, Discriminator
from train import train_gan


def run_experiment(n_epochs):
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = n_epochs
    z_dim = 64
    batch_size = 128
    lr = 0.00001

    gen = Generator(z_dim)
    disc = Discriminator()

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    dataloader = get_dataloader(batch_size)

    train_gan(gen, disc, dataloader, n_epochs, gen_opt, disc_opt, criterion, z_dim)


if __name__ == '__main__':
    run_experiment(50)
