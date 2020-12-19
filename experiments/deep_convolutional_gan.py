import torch
import torch.nn as nn
from torchvision import transforms

from dataloader import get_dataloader_mnist
from models.model_dcgan import Generator, Discriminator
from train_scripts.train_dcgan import train_dcgan
from utils.util import plot_result_after_training


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def run_experiment(n_epochs):
    # set training parameters
    criterion = nn.BCELoss()
    n_epochs = n_epochs
    z_dim = 64
    batch_size = 128
    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999

    # instantiate model
    gen = Generator(z_dim)
    disc = Discriminator()

    # instantiate optimizer
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

    # initialize model weights
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    # set dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataloader = get_dataloader_mnist(batch_size, transform)

    # start training
    train_dcgan(gen, disc, dataloader, n_epochs, gen_opt, disc_opt, criterion, z_dim)

    plot_result_after_training()
