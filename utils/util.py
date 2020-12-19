import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from utils.path_handle import result_root_path, visualization_path, create_folder


# TODO merge flatten
# TODO separate save and show
def save_tensor_images_gan(image_tensor, file_name, num_images=25, size=(1, 28, 28), show=False):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    """
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    _show_save(file_name, image_unflat, num_images, show)


def save_tensor_images_dcgan(image_tensor, file_name, num_images=25, show=False):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    """

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    _show_save(file_name, image_unflat, num_images, show)


def save_tensor_images_cgan(image_tensor, file_name, num_images=25, show=False, nrow=10):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    """

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    _show_save(file_name, image_unflat, num_images, show, nrow)


def _show_save(file_name, image_unflat, num_images, show, nrow=5):
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    image_grid = image_grid.permute(1, 2, 0).squeeze().numpy()
    if show:
        plt.imshow(image_grid)
        plt.show()
        return
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


def read_loss_from_txt(txt_path):
    with open(txt_path, "r") as f:
        str_list = f.read().splitlines()
    loss_list = [float(i) for i in str_list]
    return loss_list


def visulize_loss(discriminator_loss_path, generator_loss_path):
    discriminator_loss = read_loss_from_txt(discriminator_loss_path)
    generator_loss = read_loss_from_txt(generator_loss_path)
    plt.plot(discriminator_loss, label='Discriminator Loss')
    plt.plot(generator_loss, label='generator Loss')
    plt.legend()
    plt.savefig()


def plot_result_after_training():
    discriminator_loss_path = result_root_path / 'discriminator_loss.txt'
    generator_loss_path = result_root_path / 'generator_loss.txt'

    discriminator_loss = read_loss_from_txt(discriminator_loss_path)
    generator_loss = read_loss_from_txt(generator_loss_path)

    plt.plot(discriminator_loss, label='Discriminator Loss')
    plt.plot(generator_loss, label='generator Loss')

    plt.legend()
    plt.savefig(result_root_path / 'loss_plot.png')


def get_one_hot_labels(labels, n_classes):
    """
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    """
    return F.one_hot(labels, n_classes)


def combine_vectors(x, y):
    """
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector.
        In this assignment, this will be the noise vector of shape (n_samples, z_dim),
        but you shouldn't need to know the second dimension's size.
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    """
    # Note: Make sure this function outputs a float no matter what inputs it receives
    combined = torch.cat((x.float(), y.float()), dim=1)
    return combined


def show_tensor_images(image_tensor, num_images=16, nrow=3):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


if __name__ == '__main__':
    path = '/Users/wyh/Documents/Project/cousera/pytorch_implementation/GAN/results/discriminator_loss.txt'
    path2 = '/Users/wyh/Documents/Project/cousera/pytorch_implementation/GAN/results/generator_loss.txt'

    visulize_loss(path, path2)
