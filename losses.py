import torch

from utils.util import get_noise


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    """
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the images
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce,
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    """

    # Create noise vectors and generate a batch (num_images) of fake images.
    noise_vec = get_noise(num_images, z_dim, device)
    gen_output = gen(noise_vec).detach()

    # Get the discriminator's prediction of the fake image and calculate the loss.
    disc_output_fake = disc(gen_output)
    disc_loss_fake = criterion(disc_output_fake, torch.zeros_like(disc_output_fake))

    # Get the discriminator's prediction of the real image and calculate the loss.
    disc_output_real = disc(real)
    disc_loss_real = criterion(disc_output_real, torch.ones_like(disc_output_real))

    # Calculate the discriminator's loss by averaging the real and fake loss.
    disc_loss = (disc_loss_fake + disc_loss_real) / 2

    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    """
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the images
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce,
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    """

    # Create noise vectors and generate a batch of fake images.
    noise_vec = get_noise(num_images, z_dim, device)
    gen_output = gen(noise_vec)

    # Get the discriminator's prediction of the fake image.
    disc_output_fake = disc(gen_output)

    # Calculate the generator's loss.
    gen_loss = criterion(disc_output_fake, torch.ones_like(disc_output_fake))

    return gen_loss
