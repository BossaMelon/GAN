import torch
import torch.nn as nn
from torchvision import transforms

from dataloader import get_dataloader_mnist
from models.model_cgan import Generator, Discriminator
from train_scripts.train_cgan import train_cgan
from utils.util import plot_result_after_training


def get_input_dimensions(z_dim, mnist_shape, n_classes):
    """
    Function for getting the size of the conditional input dimensions
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
        n_classes: the total number of classes in the dataset, an integer scalar
                (10 for MNIST)
    Returns:
        generator_input_dim: the input dimensionality of the conditional generator,
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
                            (e.g. C x 28 x 28 for MNIST)
    """
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    return generator_input_dim, discriminator_im_chan


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

    # instantiate model
    mnist_shape = (1, 28, 28)
    n_classes = 10
    generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, mnist_shape, n_classes)
    gen = Generator(input_dim=generator_input_dim)
    disc = Discriminator(im_chan=discriminator_im_chan)

    # instantiate optimizer
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

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
    train_cgan(gen, disc, dataloader, n_epochs, gen_opt, disc_opt, criterion, z_dim, n_classes, mnist_shape)

    plot_result_after_training()


