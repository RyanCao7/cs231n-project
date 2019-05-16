import os

import torch
from torchvision.utils import save_image

MAX_IMG = 8
def compare_VAE(batch, recon_batch, batch_size, epoch, path):
    with torch.no_grad():
        n = min(batch.size(0), 8)
        comparison = torch.cat([batch[:n],
                               recon_batch.view(batch_size, 1, 28, 28)[:n]])
        save_image(comparison.cpu(),
                   path + '/reconstruction_' + str(epoch) + '.png', nrow=n)


def sample_VAE(vae_model, device, epoch, path):
    print('Sampling from VAE...')
    vae_model.eval()
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = vae_model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   path + '/sample_' + str(epoch) + '.png')
    print('Finished! Samples saved under ' + path + '/sample_' + str(epoch) + '.png.')