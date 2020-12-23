import torch
from torchsummary import summary
from torchvision.models import inception_v3

from models.model_controllable_gan import Generator
from utils.path_handle import pretrained_model_path
from utils.util import device


# TODO not finish!
def get_model(inception_print=False, gen_print=False):
    inception_model = inception_v3(pretrained=False)
    if inception_print:
        summary(inception_model, (3, 299, 299))
    inception_model.load_state_dict(torch.load(str(pretrained_model_path / "inception_v3_google-1a9a5a14.pth")))
    inception_model.to(device)
    inception_model = inception_model.eval()

    z_dim = 64
    gen = Generator(z_dim).to(device)
    gen_dict = torch.load(pretrained_model_path / "pretrained_celeba.pth", map_location=torch.device(device))["gen"]
    gen.load_state_dict(gen_dict)
    if gen_print:
        gen.summary()
    gen.eval()

    return inception_model, gen


def implement_fid():
    inception_model, gen = get_model()
    inception_model.fc = torch.nn.Identity()
    # summary(inception_model, (3, 299, 299))
    return inception_model


if __name__ == '__main__':
    dummy_imagebatch = torch.randn((5, 3, 299, 299))
    model = implement_fid()
    print(model(dummy_imagebatch).shape)
