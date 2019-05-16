import os
import torch
from torchvision.utils import save_image


def plot_accuracies(params):
    '''
    Simple accuracy over epoch plot.
    '''
    if not os.path.isdir('graphs/' + params['run_name'] + '/'):
        os.makedirs('graphs/' + params['run_name'] + '/')
    plt.figure(figsize=(10, 8))
    plt.title('Accuracies')
    plt.plot(params['train_accuracies'], '-o', label='Training Accuracy')
    plt.plot(params['val_accuracies'], '-o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig('graphs/' + params['run_name'] + '/' + params['run_name'] + '_accuracies.png')
    plt.close()
 

def plot_losses(params):
    '''
    Simple loss over epoch plot.
    '''
    if not os.path.isdir('graphs/' + params['run_name'] + '/'):
        os.makedirs('graphs/' + params['run_name'] + '/')
    plt.figure(figsize=(10, 8))
    plt.title('Losses')
    plt.plot(params['train_losses'], '-o', label='Training Loss')
    plt.plot(params['val_losses'], '-o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig('graphs/' + params['run_name'] + '/' + params['run_name'] + '_losses.png')
    plt.close()


MAX_IMG = 8
def compare_VAE(batch, recon_batch, batch_size, epoch, path):
    with torch.no_grad():
        n = min(batch.size(0), 8)
        comparison = torch.cat([batch[:n],
                               recon_batch.view(batch_size, 1, 28, 28)[:n]])
        save_image(comparison.cpu(),
                   path + '/reconstruction_' + str(epoch) + '.png', nrow=n)


def sample_VAE(vae_model, device, epoch, path):
    '''
    Returns randomly generated samples from vae_model.
    
    Keyword arguments:
    > vae_model (torch.nn.Module) -- generator model.
    > device (torch.device) -- CUDA or cpu.
    > epoch (int), path (string) -- for save path.
    
    Returns: N/A
    '''
    print('Sampling from VAE...')
    vae_model.eval()
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = vae_model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   path + '/sample_' + str(epoch) + '.png')
    print('Finished! Samples saved under ' + path + '/sample_' + str(epoch) + '.png.')