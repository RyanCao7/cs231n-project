import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision.utils import save_image
import constants
import train_utils


def general_plot(plot_points, plot_names, run_name, title, xlabel,
                 ylabel, plt_symbol='-o', figsize=(10, 8), scale=1):
    '''
    Generic plotting function to be used for acc/loss.
    
    Keyword arguments:
    > plot_points (list[list]) -- list of point lists to be plotted.
    > plot_names (list[string]) -- parallel list of plot names.
    > run_name (string) -- current run to save under.
    > title (string) -- plot title (also saved extension).
    > xlabel (string) -- plot x-axis label
    > ylabel (string) -- plot y-axis label
    '''
    if not os.path.isdir('graphs/' + run_name + '/'):
        os.makedirs('graphs/' + run_name + '/')
    plt.figure(figsize=figsize)
    plt.title(title)
    for idx, plot in enumerate(plot_points):
        x_labels = list(i * scale for i in range(len(plot)))
        plt.plot(x_labels, plot, plt_symbol, label=plot_names[idx])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.savefig('graphs/' + run_name + '/' + run_name + '_' + title.lower() + '.png')
    plt.close()

    
def plot_accuracies(params):
    '''
    Simple accuracy over training iterations plot.
    '''
    general_plot([params['train_accuracies'], params['val_accuracies']],
                 ['Training Accuracy', 'Validation Accuracy'], params['run_name'],
                 'Accuracies', 'Iterations', 'Accuracy', scale=params['print_frequency'])


def plot_losses(params):
    '''
    Simple loss over training iterations plot.
    '''
    general_plot([params['train_losses'], params['val_losses']],
                 ['Training Loss', 'Validation Loss'], params['run_name'],
                 'Losses', 'Iterations', 'Loss', scale=params['print_frequency'])


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
    if not os.path.isdir(path):
        os.makedirs(path)
    with torch.no_grad():
        n = min(batch.size(0), 8)
        _, recon_batch, _, _ = generator(batch)
        comparison = torch.cat([batch[:n],
                               recon_batch.view(batch.size(0), 1, 28, 28)[:n]])
        save_image(comparison.cpu(),
                   path + '/reconstruction_' + str(epoch) + '~' + constants.get_cur_time() + '.png', nrow=n)
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
                   path + '/sample_' + str(epoch) + '~' + constants.get_cur_time() + '.png')
    print('Finished! Samples saved under ' + path + '/sample_' + str(epoch) + 
          '_' + cur_time + '.png.')


def visualize_attack(params, path):
    '''
    '''
    data, perturbed_data = train_utils.sample_attack_from_dataset(params)
    save_image(data.cpu(), path + '_regular.png')
    save_image(perturbed_data.cpu(), path + '_attack.png')