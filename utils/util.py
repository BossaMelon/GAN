import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from utils.path_handle import result_root_path, visualization_path, create_folder


def save_tensor_images_gan(image_tensor, file_name, num_images=25, size=(1, 28, 28), show=False):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    """
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    _show_save(file_name, image_unflat, num_images, show)


def save_tensor_images_dcgan(image_tensor, file_name, num_images=25, size=(1, 28, 28), show=False):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    """

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    _show_save(file_name, image_unflat, num_images, show)


def _show_save(file_name, image_unflat, num_images, show):
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    image_grid = image_grid.permute(1, 2, 0).squeeze().numpy()
    if show:
        plt.imshow(image_grid)
        plt.show()

    file_path = visualization_path / '{}.jpg'.format(file_name)
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


def write_loss_to_file(loss, file_name):
    file_path = result_root_path / file_name
    with open(file_path, "a+") as file:
        file.write(f"{loss:.4f}" + '\n')


def write_traininfo_to_file(info):
    file_path = result_root_path / 'train_info.txt'
    with open(file_path, "rb") as file:
        file.write(info)


create_folder()

if __name__ == '__main__':
    write_loss_to_file(22.4, 'dis.txt')
    write_loss_to_file(222, 'gen.txt')
