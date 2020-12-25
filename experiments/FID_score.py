import sys

import numpy as np
import scipy.linalg
import torch
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.models import inception_v3
from tqdm.auto import tqdm

sys.path.append('..')
from dataloader import get_dataloader_celebA
from models.model_controllable_gan import Generator
from utils.path_handle import pretrained_model_path
from utils.util import device, get_noise

z_dim = 64
batch_size = 4  # Samples per iteration


# TODO not finish!
def get_model(inception_print=False, gen_print=False):
    inception_model = inception_v3(pretrained=False, init_weights=False)
    if inception_print:
        summary(inception_model, (3, 299, 299))
    inception_model.load_state_dict(torch.load(str(pretrained_model_path / "inception_v3_google-1a9a5a14.pth")))
    inception_model.to(device)
    inception_model = inception_model.eval()

    gen = Generator(z_dim).to(device)
    gen_dict = torch.load(pretrained_model_path / "pretrained_celeba.pth", map_location=torch.device(device))["gen"]
    gen.load_state_dict(gen_dict)
    if gen_print:
        gen.summary()
    gen.eval()

    return inception_model, gen


def _get_dataset():
    image_size = 299

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = CelebA(".", download=True, transform=transform)
    return dataset


def get_dataloader():
    image_size = 299

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataloader = get_dataloader_celebA(batch_size=batch_size, transform=transform)

    return dataloader


def matrix_sqrt(x):
    """
    Function that takes in a matrix and returns the square root of that matrix.
    For an input matrix A, the output matrix B would be such that B @ B is the matrix A.
    Parameters:
        x: a matrix
    """
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)


def preprocess(img):
    img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
    return img


def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    """
    Function for returning the Fréchet distance between multivariate Gaussians,
    parameterized by their means and covariance matrices.
    Parameters:
        mu_x: the mean of the first Gaussian, (n_features)
        mu_y: the mean of the second Gaussian, (n_features)
        sigma_x: the covariance matrix of the first Gaussian, (n_features, n_features)
        sigma_y: the covariance matrix of the second Gaussian, (n_features, n_features)
    """
    res = torch.norm(mu_x - mu_y) ** 2 + torch.trace(sigma_x + sigma_y - 2 * matrix_sqrt(sigma_x @ sigma_y))
    return res


def extract_feature_real_fake():
    inception_model, gen = get_model()
    inception_model.fc = torch.nn.Identity()
    # summary(inception_model, (3, 299, 299))

    fake_features_list = []
    real_features_list = []

    gen.eval()
    n_samples = 512  # The total number of samples

    dataloader = get_dataloader()

    cur_samples = 0
    with torch.no_grad():  # You don't need to calculate gradients here, so you do this to save memory
        try:
            for real_example, _ in tqdm(dataloader, total=n_samples // batch_size):  # Go by batch
                real_samples = real_example
                real_features = inception_model(real_samples.to(device)).detach().to('cpu')  # Move features to CPU
                real_features_list.append(real_features)

                fake_samples = get_noise(len(real_example), z_dim).to(device)
                fake_samples = preprocess(gen(fake_samples))
                fake_features = inception_model(fake_samples.to(device)).detach().to('cpu')
                fake_features_list.append(fake_features)
                cur_samples += len(real_samples)
                if cur_samples >= n_samples:
                    break
        except:
            print("Error in loop")

    return fake_features_list, real_features_list


def caculate_fid(fake_features_list, real_features_list):
    fake_features_all = torch.cat(fake_features_list)
    real_features_all = torch.cat(real_features_list)
    mu_fake = torch.mean(fake_features_all, dim=0)
    mu_real = torch.mean(real_features_all, dim=0)
    sigma_fake = get_covariance(fake_features_all)
    sigma_real = get_covariance(real_features_all)
    with torch.no_grad():
        print(frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item())


def run():
    fake_features_list, real_features_list = extract_feature_real_fake()
    caculate_fid(fake_features_list, real_features_list)


if __name__ == '__main__':
    # dummy_imagebatch = torch.randn((5, 3, 299, 299))
    run()
