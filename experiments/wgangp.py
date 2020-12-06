import torch
import torch.nn as nn
from torchvision import transforms

from dataloader import get_dataloader
from models.model_wgangp import Generator, Critic
from train_scripts.train_wgangp import train_wgangp
from utils.util import plot_result_after_training


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def run_experiment(n_epochs):
    # set training parameters
    n_epochs = n_epochs
    z_dim = 64
    batch_size = 128
    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999
    c_lambda = 10
    crit_repeats = 5

    # instantiate model
    gen = Generator(z_dim)
    disc = Critic()

    # instantiate optimizer
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

    # initialize model weights
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    # set dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataloader = get_dataloader(batch_size, transform)

    # start training
    train_wgangp(gen, disc, dataloader, n_epochs, gen_opt, disc_opt, z_dim, c_lambda)

    plot_result_after_training()
