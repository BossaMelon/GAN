import torch
from tqdm.auto import tqdm

from utils.util import write_loss_to_file, get_noise, save_tensor_images_cgan, get_one_hot_labels, combine_vectors

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_name = 'cpu' if not torch.cuda.is_available() else torch.cuda.get_device_name()


def train_cgan(gen, disc, dataloader, epochs, gen_opt, disc_opt, criterion, z_dim, n_classes, mnist_shape):
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

        gen.train()
        for real, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs - 1}"):
            cur_batch_size = len(real)

            real = real.to(device)
            one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

            # Update discriminator
            disc_opt.zero_grad()

            fake_noise = get_noise(cur_batch_size, z_dim, device=device)

            noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
            fake = gen(noise_and_labels)

            fake_image_and_labels = torch.cat((fake, image_one_hot_labels.float()), dim=1)
            real_image_and_labels = torch.cat((real, image_one_hot_labels.float()), dim=1)
            disc_fake_pred = disc(fake_image_and_labels)
            disc_real_pred = disc(real_image_and_labels)

            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Keep track of the epoch sum discriminator loss
            discriminator_loss += disc_loss.item() * cur_batch_size

            # Update generator
            gen_opt.zero_grad()
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            # This will error if you didn't concatenate your labels to your image correctly
            disc_fake_pred = disc(fake_image_and_labels)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the epoch sum generator loss
            generator_loss += gen_loss.item() * cur_batch_size

        mean_discriminator_loss = discriminator_loss / data_size
        mean_generator_loss = generator_loss / data_size

        write_loss_to_file(mean_discriminator_loss, 'discriminator_loss.txt')
        write_loss_to_file(mean_generator_loss, 'generator_loss.txt')

        print(f"Generator loss: {mean_generator_loss:.4f}     discriminator loss: {mean_discriminator_loss:.4f}")

        # Visualization/validation
        labels = [i % 10 for i in range(100)]
        labels = torch.as_tensor(labels, dtype=torch.long)
        fake_noise = get_noise(n_samples=100, z_dim=z_dim, device=device)
        one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
        noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
        gen.eval()
        fake = gen(noise_and_labels)
        save_tensor_images_cgan(fake, f'cgan-{epoch + 1}', num_images=100)


def eval_cgan():
    pass


if __name__ == '__main__':
    labels = [i % 10 for i in range(100)]
    print(labels)
