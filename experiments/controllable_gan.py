import getpass

import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torchvision import transforms
from tqdm.auto import tqdm

from dataloader import get_dataloader_celebA
from models.model_controllable_gan import Generator, Classifier, device, device_name
from utils.util import get_noise, save_tensor_images_dcgan, show_tensor_images

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_name = 'cpu' if not torch.cuda.is_available() else torch.cuda.get_device_name()
z_dim = 64

feature_names = ["5oClockShadow", "ArchedEyebrows", "Attractive", "BagsUnderEyes", "Bald", "Bangs",
                 "BigLips", "BigNose", "BlackHair", "BlondHair", "Blurry", "BrownHair", "BushyEyebrows", "Chubby",
                 "DoubleChin", "Eyeglasses", "Goatee", "GrayHair", "HeavyMakeup", "HighCheekbones", "Male",
                 "MouthSlightlyOpen", "Mustache", "NarrowEyes", "NoBeard", "OvalFace", "PaleSkin", "PointyNose",
                 "RecedingHairline", "RosyCheeks", "Sideburn", "Smiling", "StraightHair", "WavyHair",
                 "WearingEarrings",
                 "WearingHat", "WearingLipstick", "WearingNecklace", "WearingNecktie", "Young"]
n_images = 8
fake_image_history = []
grad_steps = 20  # Number of gradient steps to take
skip = 2  # Number of gradient steps to skip in the visualization


# TODO check how to train_generator? Simply train with DCGAN?
def train_generator():
    pass


# TODO merge to path handle
def load_pretrained_models():
    current_username = getpass.getuser()
    if current_username == 'wyh':
        model_root = '/Users/wyh/Documents/Project/cousera/pytorch_implementation/GAN/trained_models'
    elif current_username == 'wangyueh':
        model_root = ''
    else:
        raise Exception('not defined path')

    gen = Generator(z_dim).to(device)
    gen_dict = torch.load(model_root + "/pretrained_celeba.pth", map_location=torch.device(device))["gen"]
    gen.load_state_dict(gen_dict)
    gen.eval()

    n_classes = 40
    classifier = Classifier(n_classes=n_classes).to(device)
    class_dict = torch.load(model_root + "/pretrained_classifier.pth", map_location=torch.device(device))["classifier"]
    classifier.load_state_dict(class_dict)
    classifier.eval()
    print("Loaded the models!")

    opt = torch.optim.Adam(classifier.parameters(), lr=0.01)
    return gen, classifier, opt


def train_classifier(filename):
    """
    indices stands for whether in the pic there is:
      5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair
      Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones
      Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline
      Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick
      Wearing_Necklace Wearing_Necktie Young
    """

    # Target all the classes, so that's how many the classifier will learn
    # can also train part of the labels eg. label_indices = (1, 2, 3)
    label_indices = range(40)

    n_epochs = 3
    display_step = 10
    lr = 0.001
    beta_1 = 0.5
    beta_2 = 0.999
    image_size = 64
    batch_size = 128

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataloader = get_dataloader_celebA(batch_size, transform)

    classifier = Classifier(n_classes=len(label_indices)).to(device)
    class_opt = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(beta_1, beta_2))
    criterion = nn.BCELoss()

    cur_step = 0
    classifier_losses = []
    # classifier_val_losses = []
    print()
    print(f'Start training on {device_name}')
    print(64 * '-')

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for real, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs - 1}"):
            real = real.to(device)
            labels = labels[:, label_indices].to(device).float()

            class_opt.zero_grad()
            class_pred = classifier(real)
            class_loss = criterion(class_pred, labels)
            class_loss.backward()  # Calculate the gradients
            class_opt.step()  # Update the weights
            classifier_losses.append(class_loss.item())  # Keep track of the average classifier loss

            # Visualization code
            if cur_step % display_step == 0 and cur_step > 0:
                class_mean = sum(classifier_losses[-display_step:]) / display_step
                print(f"Step {cur_step}: Classifier loss: {class_mean}")
                # step_bins = 20
                # x_axis = sorted([i * step_bins for i in range(len(classifier_losses) // step_bins)] * step_bins)
                # sns.lineplot(x_axis, classifier_losses[:len(x_axis)], label="Classifier Loss")
                # plt.legend()
                # plt.show()
                torch.save({"classifier": classifier.state_dict()}, filename)
            cur_step += 1


def test_generator():
    n_images = 25
    gen, _, _ = load_pretrained_models()
    noise = get_noise(n_images, z_dim).to(device)
    fake = gen(noise)
    save_tensor_images_dcgan(fake, '', show=True)
    print()


def _calculate_updated_noise(noise, weight):
    """
    Function to return noise vectors updated with stochastic gradient ascent.
    Parameters:
        noise: the current noise vectors. You have already called the backwards function on the target class
          so you can access the gradient of the output class with respect to the noise by using noise.grad
        weight: the scalar amount by which you should weight the noise gradient
    """
    new_noise = noise + (noise.grad * weight)
    return new_noise


def run_conditional_gen():
    gen, classifier, opt = load_pretrained_models()

    ### Change me! ###
    target_indices = feature_names.index("Male")  # Feel free to change this value to any string from feature_names!

    noise = get_noise(n_images, z_dim).to(device).requires_grad_()
    for i in range(grad_steps):
        opt.zero_grad()
        fake = gen(noise)
        fake_image_history.append(fake)
        fake_classes_score = classifier(fake)[:, target_indices].mean()
        fake_classes_score.backward()
        noise.data = _calculate_updated_noise(noise, 1 / grad_steps)

    plt.rcParams['figure.figsize'] = [n_images * 2, grad_steps * 2 / skip]
    show_tensor_images(torch.cat(fake_image_history[::skip], dim=2), num_images=n_images, nrow=n_images)


def _get_score(current_classifications, original_classifications, target_indices, other_indices, penalty_weight):
    """
    Function to return the score of the current classifications, penalizing changes
    to other classes with an L2 norm.
    Parameters:
        current_classifications: the classifications associated with the current noise
        original_classifications: the classifications associated with the original noise
        target_indices: the index of the target class
        other_indices: the indices of the other classes
        penalty_weight: the amount that the penalty should be weighted in the overall score
    """
    # Steps: 1) Calculate the change between the original and current classifications (as a tensor)
    #           by indexing into the other_indices you're trying to preserve, like in x[:, features].
    #        2) Calculate the norm (magnitude) of changes per example.
    #        3) Multiply the mean of the example norms by the penalty weight.
    #           This will be your other_class_penalty.
    #           Make sure to negate the value since it's a penalty!
    #        4) Take the mean of the current classifications for the target feature over all the examples.
    #           This mean will be your target_score.
    # Calculate the norm (magnitude) of changes per example and multiply by penalty weight
    other_class_penalty = -torch.norm((current_classifications - original_classifications)[:, other_indices],
                                      dim=1).mean() * penalty_weight
    # Take the mean of the current classifications for the target feature
    target_score = current_classifications[:, target_indices].mean()
    return target_score + other_class_penalty


def run_conditional_gen_with_regularization():
    gen, classifier, opt = load_pretrained_models()

    fake_image_history = []
    ### Change me! ###
    target_indices = feature_names.index(
        "Eyeglasses")  # Feel free to change this value to any string from feature_names from earlier!
    other_indices = [cur_idx != target_indices for cur_idx, _ in enumerate(feature_names)]
    noise = get_noise(n_images, z_dim).to(device).requires_grad_()
    original_classifications = classifier(gen(noise)).detach()
    for i in range(grad_steps):
        opt.zero_grad()
        fake = gen(noise)
        fake_image_history += [fake]
        fake_score = _get_score(
            classifier(fake),
            original_classifications,
            target_indices,
            other_indices,
            penalty_weight=0.1
        )
        fake_score.backward()
        noise.data = _calculate_updated_noise(noise, 1 / grad_steps)

    plt.rcParams['figure.figsize'] = [n_images * 2, grad_steps * 2]
    show_tensor_images(torch.cat(fake_image_history[::skip], dim=2), num_images=n_images, nrow=n_images)


if __name__ == '__main__':
    run_conditional_gen_with_regularization()
