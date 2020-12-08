import torch
from tqdm.auto import tqdm

from losses.wgangp_losses import get_crit_loss, get_gen_loss, get_gradient, gradient_penalty
from train import device, device_name
from utils.util import write_loss_to_file, get_noise, save_tensor_images_dcgan

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_name = 'cpu' if not torch.cuda.is_available() else torch.cuda.get_device_name()


def train_wgangp(gen, crit, dataloader, epochs, gen_opt, crit_opt, z_dim, c_lambda):
    gen = gen.to(device)
    crit = crit.to(device)
    crit_repeats = 5
    data_size = len(dataloader.dataset)
    critic_losses = []
    generator_losses = []

    print()
    print(f'Start training on {device_name}')
    print(64 * '-')

    for epoch in range(epochs):
        generator_loss = 0.
        discriminator_loss = 0.
        # Dataloader returns the batches

#        for real, _ in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs - 1}"):
        for real, _ in dataloader:

            cur_batch_size = len(real)

            real = real.to(device)
            mean_iteration_critic_loss = 0

            for _ in range(crit_repeats):
                # Update discriminator
                crit_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, z_dim, device)
                fake = gen(fake_noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)

                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(crit, real, fake.detach(), epsilon)

                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()
            critic_losses.append(mean_iteration_critic_loss)
            print(f"C: {mean_iteration_critic_loss}")

            # Update generator
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            fake_2 = gen(fake_noise_2)
            crit_fake_pred = crit(fake_2)

            gen_loss = get_gen_loss(crit_fake_pred)
            gen_loss.backward()

            # Update the weights
            gen_opt.step()

            # Keep track of the average generator loss
            generator_losses.append(gen_loss.item())
            print(f'G: {gen_loss.item()}')

        mean_critic_loss = sum(critic_losses)
        mean_generator_loss = sum(generator_losses)
        print(mean_critic_loss)
        print(mean_generator_loss)

        write_loss_to_file(mean_critic_loss, 'discriminator_loss.txt')
        write_loss_to_file(mean_generator_loss, 'generator_loss.txt')

        print(f"Generator loss: {mean_generator_loss:.4f}     discriminator loss: {mean_critic_loss:.4f}")

        # Visualization
        fake_noise = get_noise(64, z_dim, device=device)
        fake = gen(fake_noise)
        save_tensor_images_dcgan(fake, f'wgan-{epoch + 1}')
