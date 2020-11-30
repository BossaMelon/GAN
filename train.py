import torch
import torch.nn as nn
from tqdm.auto import tqdm
from loss import get_gen_loss, get_disc_loss
from util import show_tensor_images, get_noise

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(gen, disc, dataloader, n_epochs, gen_opt, disc_opt, criterion, z_dim,display_step=500):
    gen = gen.to(device)
    disc = disc.to(device)
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    gen_loss = False
    error = False

    for epoch in range(n_epochs):

        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)

            # Flatten the batch of real images from the dataset
            real = real.view(cur_batch_size, -1).to(device)

            ### Update discriminator ###
            # Zero out the gradients before backpropagation
            disc_opt.zero_grad()

            # Calculate discriminator loss
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Update optimizer
            disc_opt.step()

            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###
            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                show_tensor_images(fake)
                show_tensor_images(real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1


if __name__ == '__main__':
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.00001
