import torch
import torch.nn as nn

from dataloader import get_dataloader_mnist
from models.model_gan import Generator, Discriminator
from train_scripts.train_gan import train_gan
from utils.util import plot_result_after_training


def run_experiment(n_epochs):
    criterion = nn.BCELoss()
    n_epochs = n_epochs
    z_dim = 64
    batch_size = 128
    lr = 0.00001

    gen = Generator(z_dim)
    disc = Discriminator()

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    dataloader = get_dataloader_mnist(batch_size)

    train_gan(gen, disc, dataloader, n_epochs, gen_opt, disc_opt, criterion, z_dim)

    plot_result_after_training()



