import glob
import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision.utils import save_image
import constants
import train_utils
import box_utils
import threading


def save_and_upload_image(tensor, save_path, **kwargs):
    '''
    Wrapper to save and upload an image.
    '''
    rid_duplicate(save_path)
    save_image(tensor, save_path, **kwargs)
    box_utils.upload_single(save_path)


def rid_duplicate(save_path):
    '''
    Checks to see if there are any older duplicates, and removes
    them if so.
    
    Keyword arguments:
    save_path (string) -- full path of the file to be saved.
    '''
    possible_duplicates_name = save_path[:save_path.rfind('~')]
    directory = save_path[save_path.find('/'):]
    
    # If a previous version exists, get rid of it.
    for name in glob.glob(directory + '*'):
        if possible_duplicates_name in name:
            subprocess.check_output(['rm', name])


def general_plot(plot_points, plot_names, run_name, title, xlabel,
                 ylabel, plt_symbol='-o', figsize=(10, 8), scales=[1]):
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
    save_name = 'graphs/' + run_name + '/' + run_name + '_' + title.lower() + \
                '~' + constants.get_cur_time() + '.png'
    if not os.path.isdir('graphs/' + run_name + '/'):
        os.makedirs('graphs/' + run_name + '/')
    plt.figure(figsize=figsize)
    plt.title(title)
    for idx, plot in enumerate(plot_points):
        x_labels = list(i * scales[idx] for i in range(len(plot)))
        plt.plot(x_labels, plot, plt_symbol, label=plot_names[idx])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.savefig(save_name)
    plt.close()
    
    # Syncs with Box
    rid_duplicate(save_name)
    box_utils.upload_single(save_name)

    
def plot_accuracies(params):
    '''
    Simple accuracy over training iterations plot.
    '''
    train_scale = params['print_frequency'] / (params['batch_size'] * len(params['train_dataloader']))
    val_scale = params['print_frequency'] / (params['batch_size'] * len(params['val_dataloader']))
    general_plot([params['train_accuracies'], params['val_accuracies']],
                 ['Training Accuracy', 'Validation Accuracy'], params['run_name'],
                 'Accuracies', 'Iterations', 'Accuracy', scales=[train_scale, val_scale])


def plot_losses(params):
    '''
    Simple loss over training iterations plot.
    '''
    train_scale = params['print_frequency'] / (params['batch_size'] * len(params['train_dataloader']))
    val_scale = params['print_frequency'] / (params['batch_size'] * len(params['val_dataloader']))
    general_plot([params['train_losses'], params['val_losses']],
                 ['Training Loss', 'Validation Loss'], params['run_name'],
                 'Losses', 'Iterations', 'Loss', scales=[train_scale, val_scale])


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
    save_path = path + '/reconstruction_' + str(epoch) + '~' + constants.get_cur_time() + '.png'
    with torch.no_grad():
        n = min(batch.size(0), 8)
        _, recon_batch, _, _ = generator(batch)
        comparison = torch.cat([batch[:n],
                               recon_batch.view(batch.size(0), 1, 28, 28)[:n]])
        save_and_upload_image(comparison.cpu(), save_path, nrow=n)

    print('Finished! Samples saved under ' + save_path)

    
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
    save_path = path + '/sample_' + str(epoch) + '~' + constants.get_cur_time() + '.png'
    vae_model.eval()
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = vae_model.decode(sample).cpu()
        save_and_upload_image(sample.view(64, 1, 28, 28), save_path)

    print('Finished! Samples saved under ' + save_path)
    

def visualize_attack(params, path):
    '''
    '''
    data, perturbed_data = train_utils.sample_attack_from_dataset(params)
    print(data.min(), data.max(), perturbed_data.min(), perturbed_data.max())
    save_and_upload_image(data.cpu(), path + '_regular.png')
    save_and_upload_image(perturbed_data.cpu(), path + '_attack.png')
    save_and_upload_image(data.cpu(), path + '_regular_norm.png', normalize=True)
    save_and_upload_image(perturbed_data.cpu(), path + '_attack_norm.png', normalize=True)
