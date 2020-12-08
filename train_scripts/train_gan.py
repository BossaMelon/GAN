import torch
from tqdm.auto import tqdm

from losses.gan_losses import get_disc_loss, get_gen_loss
from utils.util import write_loss_to_file, get_noise, save_tensor_images_gan

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_name = 'cpu' if not torch.cuda.is_available() else torch.cuda.get_device_name()


def train_gan(gen, disc, dataloader, epochs, gen_opt, disc_opt, criterion, z_dim):
    gen = gen.to(device)
    disc = disc.to(device)

    data_size = len(dataloader.dataset)

    print()
    print(f'Start training on {device_name}')
    print(64 * '-')

    for epoch in range(epochs):
        generator_loss = 0.
        discriminator_loss = 0.
        # Dataloader returns the batches

        for real, _ in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs - 1}"):
            cur_batch_size = len(real)

            # Flatten the batch of real images from the dataset
            real = real.view(cur_batch_size, -1).to(device)

            # Update discriminator
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Update generator
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the epoch sum discriminator loss
            discriminator_loss += disc_loss.item() * cur_batch_size

            # Keep track of the epoch sum generator loss
            generator_loss += gen_loss.item() * cur_batch_size

        mean_discriminator_loss = discriminator_loss / data_size
        mean_generator_loss = generator_loss / data_size

        write_loss_to_file(mean_discriminator_loss, 'discriminator_loss.txt')
        write_loss_to_file(mean_generator_loss, 'generator_loss.txt')

        print(f"Generator loss: {mean_generator_loss:.4f}     discriminator loss: {mean_discriminator_loss:.4f}")

        # Visualization
        fake_noise = get_noise(64, z_dim, device=device)
        fake = gen(fake_noise)
        save_tensor_images_gan(fake, f'gan-{epoch + 1}')
