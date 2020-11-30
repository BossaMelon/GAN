import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

image_path = data_root = './visualization'


def save_tensor_images(image_tensor, file_name, num_images=25, size=(1, 28, 28), show=False):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    """
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    image_grid = image_grid.permute(1, 2, 0).squeeze().numpy()

    if show:
        plt.imshow(image_grid)
        plt.show()

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    file_path = image_path + '/{}.jpg'.format(file_name)
    plt.imsave(file_path, image_grid)


def get_noise(n_samples, z_dim, device='cpu'):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    """
    return torch.randn(n_samples, z_dim, device=device)

