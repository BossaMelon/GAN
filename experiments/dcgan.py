import torch
import torch.nn as nn
from torchvision import transforms

from dataloader import get_dataloader
from models.model_dcgan import Generator, Discriminator
from train import train_dcgan


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def main():
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    z_dim = 64
    batch_size = 128
    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999

    gen = Generator(z_dim)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc = Discriminator()
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataloader = get_dataloader(batch_size, transform)

    train_dcgan(gen, disc, dataloader, n_epochs, gen_opt, disc_opt, criterion, z_dim)


if __name__ == '__main__':
    main()