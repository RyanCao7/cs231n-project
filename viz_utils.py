import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision.utils import save_image


# TODO: Massive plot refactoring! Need a single general plotting function!
def plot_accuracies(params):
    '''
    Simple accuracy over epoch plot. TODO: MAKE THIS PLOT PER N BATCHES!
    '''
    if not os.path.isdir('graphs/' + params['run_name'] + '/'):
        os.makedirs('graphs/' + params['run_name'] + '/')
    plt.figure(figsize=(10, 8))
    plt.title('Accuracies')
    plt.plot(params['train_accuracies'], '-o', label='Training Accuracy')
    plt.plot(params['val_accuracies'], '-o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    extension = '_accuracies.png'
    plt.savefig('graphs/' + params['run_name'] + '/' + params['run_name'] + extension)
    plt.close()


def plot_losses(params):
    '''
    Simple loss over epoch plot. TODO: MAKE THIS PLOT PER N BATCHES!
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
def compare_VAE(batch, generator, epoch, path):
    '''
    Saves parallely generated batch/reconstructed batch
    images next to each other.
    
    Keyword arguments:
    > batch (torch.tensor) -- original input data.
    > generator (torch.nn.Module) -- current generator.
    > epoch (int) -- number of COMPLETED training epochs.
    > path (string) -- save path.
    
    Returns: N/A
    '''
    print('Generating comparison images from VAE...')
    cur_time = datetime.now().strftime("%m-%d-%Y~%H_%M_%S")
    if not os.path.isdir(path):
        os.makedirs(path)
    with torch.no_grad():
        n = min(batch.size(0), 8)
        _, recon_batch, _, _ = generator(batch)
        comparison = torch.cat([batch[:n],
                               recon_batch.view(batch.size(0), 1, 28, 28)[:n]])
        save_image(comparison.cpu(),
                   path + '/reconstruction_' + str(epoch) + '_' + cur_time + '.png', nrow=n)
    print('Finished! Samples saved under ' + path + '/reconstruction_' + str(epoch) + '_' + 
          cur_time + '.png.')

    
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
    cur_time = datetime.now().strftime("%m-%d-%Y~%H_%M_%S")
    if not os.path.isdir(path):
        os.makedirs(path)
    vae_model.eval()
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = vae_model.decode(sample).cpu()
        
        save_image(sample.view(64, 1, 28, 28),
                   path + '/sample_' + str(epoch) + '_' + cur_time + '.png')
    print('Finished! Samples saved under ' + path + '/sample_' + str(epoch) + 
          '_' + cur_time + '.png.')