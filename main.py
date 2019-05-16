import argparse
import os
import random
import shutil
import time
import warnings
import sys
import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# For DataLoader and transforms
from data_utils import get_dataloader

# For training utility functions
import train_utils

# For models
import models

# For constants
import constants

# For adversarial batch generation
import adversary


def new_model(params):
    '''
    Creates a new model instance and initializes the respective fields
    in params.

    Keyword arguments:
    > params (dict) -- current state variable

    Returns: N/A
    '''

    params = param_factory()
    
    # Name
    params['run_name'] = input('Please type the current model run name -> ')

    # Architecture
    model_string = train_utils.input_from_list(constants.MODEL_ARCHITECTURE_NAMES, 'model')
    if model_string == 'Classifier_A':
        params['model'] = models.Classifier_A()
    elif model_string == 'Classifier_B':
        params['model'] = models.Classifier_B()
    if model_string == 'Classifier_C':
        params['model'] = models.Classifier_C()
    elif model_string == 'Classifier_D':
        params['model'] = models.Classifier_D()
    elif model_string == 'Classifier_E':
        params['model'] = models.Classifier_E()
    elif model_string == 'VAE':
        params['model'] = models.VAE()
    models.initialize_model(params['model'])

    # Setup other state variables
    for state_var in constants.SETUP_STATE_VARS:
        train_utils.store_user_choice(params, state_var)
        print()

    # TODO: Allow user to pick criteria!!!

    # Optimizer - TODO: Allow user to pick optimizer
    params['optimizer'] = torch.optim.SGD(params['model'].parameters(), params['learning_rate'],
                                momentum=params['momentum'],
                                weight_decay=params['weight_decay'])

    # Grabs dataloaders. TODO: Prompt for val split/randomize val indices
    params['train_dataloader'], params['val_dataloader'], params['test_dataloader'] = get_dataloader(
        dataset_name=params['dataset'], 
        batch_sz=params['batch_size'],
        num_threads=params['num_threads'])

    # Saves an initial copy
    if not os.path.isdir('models/' + params['run_name']):
        os.makedirs('models/' + params['run_name'])
    train_utils.save_checkpoint(params, 0)


def load_model(params):
    '''
    Loads a model from a given checkpoint.
    '''
    # Grabs model directory from user
    model_folders = glob.glob('models/*')
    if len(model_folders) == 0:
        print('No current models exist. Switching to creating a new model...')
        new_model(params)
        return

    # Grabs model choice from user
    print('\n --- All saved models ---')
    user_model_choice = train_utils.input_from_list(model_folders, 'input')
    print('Chosen model:', user_model_choice[user_model_choice.rfind('/') + 1:])

    # Grabs checkpoint file from user
    print('\n --- All saved checkpoints ---')
    saved_checkpoint_files = glob.glob(user_model_choice + '/*')
    user_checkpoint_choice = train_utils.input_from_list(saved_checkpoint_files, 'checkpoint')
    print('Chosen checkpoint:', user_checkpoint_choice[user_checkpoint_choice.rfind('/') + 1:], '\n')

    # Loads saved state and sets up GPU for its model
    print('Loading model...')
    loaded = torch.load(user_checkpoint_choice)
    print('Finished loading model! Use -p to print current state.')
    return loaded


def setup_cuda(params):
    '''
    Loads model onto GPU if one is available.
    '''
    print("Use {} for training".format(params['device']))
    params['model'] = params['model'].to(params['device'])

    # Should make things faster if input size is consistent.
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
    cudnn.benchmark = True


def param_factory():
    '''
    Constructs a default parameter dictionary to be loaded up 
    upon start of console program.

    Keyword arguments: N/A

    Return value: params
    > params (dict) -- dictionary of default parameters.
    '''
    params = {}
    params['dataset'] = 'MNIST'
    params['model'] = None
    params['load_workers'] = 4
    params['total_epochs'] = 90
    params['cur_epoch'] = 0
    params['batch_size'] = 256
    params['learning_rate'] = 0.1
    params['momentum'] = 0.9
    params['weight_decay'] = 1e-4
    params['print_frequency'] = 10
    params['best_val_acc'] = 0.
    params['num_threads'] = 4

    # Default - checkpoint every 10 epochs
    params['save_every'] = 10

    # For each run to have a unique identifier
    params['run_name'] = None

    # Whether to evaluate on validation set
    params['evaluate'] = True

    # Random seed
    params['seed'] = None

    # For GPU forwarding
    params['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # define loss function (criterion) and optimizer
    params['criterion'] = nn.CrossEntropyLoss().to(params['device'])

    # Optimizer - THIS MUST GET CONSTRUCTED LATER
    params['optimizer'] = None

    # Dataloaders
    params['train_dataloader'] = None
    params['val_dataloader'] = None
    params['test_dataloader'] = None

    # Saved per-epoch accuracies + losses
    params['train_accuracies'] = []
    params['train_losses'] = []
    params['val_accuracies'] = []
    params['val_losses'] = []

    return params


def edit_state_vals(params):
    editable_vars_with_vals = [state_var + ': ' + str(params[state_var]) for state_var in constants.EDITABLE_STATE_VARS]
    editable_vars_with_vals.append('exit')
    return editable_vars_with_vals


def edit_state(params):
    '''
    Allows user to edit current state parameters.

    Keyword arguments:
    > params (dict) -- current state variable

    Returns: N/A
    '''

    if params['model'] is None:
        print('No model loaded! Type -n to create a new model, or -l to load an existing one from file.\n')
        return

    print('--- Editing current state ---')
    editable_vars_with_vals = edit_state_vals(params)
    change_var = train_utils.input_from_list(editable_vars_with_vals, 'state variable')
    if change_var != 'exit':
        change_var = change_var[:change_var.rfind(':')]
    while change_var != 'exit':
        train_utils.store_user_choice(params, change_var)
        editable_vars_with_vals = edit_state_vals(params)
        change_var = train_utils.input_from_list(editable_vars_with_vals, 'state variable')
        if change_var != 'exit':
            change_var = change_var[:change_var.rfind(':')]

    # TODO: I don't know if this is entirely correct. Think about it!
    train_utils.save_checkpoint(params, params['cur_epoch'])
    print()


# TODO: Actually print useful stuff :P
def print_state(params):
    '''
    Nicely formats and prints the currently loaded state.

    Keyword arguments: params
    > params (dict) -- currently loaded state dict.
    '''
    print('\n --- Loaded state --- \n')
    print('Model:', params['model'], '\n')
    print('Optimizer:', params['optimizer'], '\n')
    print('Device:', params['device'])
    print('Epoch:', params['cur_epoch'])
    print('Total epochs:', params['total_epochs'])
    print('Batch size:', params['batch_size'])
    if params['train_dataloader'] is not None:
        print('Total training set size:', params['batch_size'] * len(params['train_dataloader']))
        print('Total val set size:', params['batch_size'] * len(params['val_dataloader']))
        print('Total test set size:', params['batch_size'] * len(params['test_dataloader']))
    print('Print every', params['print_frequency'], 'iterations.')
    print('Save every', params['save_every'], 'epochs.')
    print()
    

def perform_training(params, evaluate=False):
    '''
    Attempts to train the remaining number of epochs. Will fail
    if no valid model is loaded.

    Keyword arguments: params
    > params (dict) -- currently loaded state dict.

    Returns: N/A
    '''
    if params['model'] is None:
        print('No model loaded! Type -n to create a new model, or -l to load an existing one from file.\n')
        return

    setup_cuda(params)
    
    # Training/val loop
    for epoch in range(params['cur_epoch'] + 1, params['total_epochs'] + 1):
        
        print('Training: begin epoch', epoch)

        # LR Decay - currently a stepwise decay
        adjust_learning_rate(epoch, params)

        # train for one epoch
        train_one_epoch(epoch, params)

        # evaluate on validation set
        acc1 = validate(params)

        # Update best val accuracy
        params['best_val_acc'] = max(acc1, params['best_val_acc'])

        # Save checkpoint every 'save_every' epochs.
        if epoch % params['save_every'] == 0:
            train_utils.save_checkpoint(params, epoch)

        # Update the current epoch
        params['cur_epoch'] += 1

        # Update train/val accuracy/loss plots
        train_utils.plot_accuracies(params)
        train_utils.plot_losses(params)

    train_utils.save_checkpoint(params, params['total_epochs'])


def train_one_epoch(epoch, params):

    # No idea what this is for now...
    batch_time = train_utils.AverageMeter('Time', ':5.3f')
    data_time = train_utils.AverageMeter('Data', ':5.3f')
    losses = train_utils.AverageMeter('Loss', ':.4e')
    top1 = train_utils.AverageMeter('Acc@1', ':5.2f')
    top5 = train_utils.AverageMeter('Acc@5', ':5.2f')
    progress = train_utils.ProgressMeter(len(params['train_dataloader']), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # Switch to train mode. Important for dropout and batchnorm.
    params['model'].train()

    end = time.time()
    for i, (data, target) in enumerate(params['train_dataloader']):
        # measure data loading time
        data_time.update(time.time() - end)

        data = data.to(params['device'])
        target = target.to(params['device'])

        # compute output
        output = params['model'](data)
        loss = params['criterion'](output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))

        # compute gradient and do SGD step
        params['optimizer'].zero_grad()
        loss.backward()
        params['optimizer'].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print training progress
        if i % params['print_frequency'] == 0:
            progress.print(i)

    # Storing training losses/accuracies
    params['train_losses'].append(losses.get_avg())
    params['train_accuracies'].append(top1.get_avg())


def validate(params, adversarial=False):

    if params['model'] is None:
        print('No model loaded! Type -n to create a new model, or -l to load an existing one from file.\n')
        return
    
    print('--- BEGIN VALIDATION PASS ---')
    
    setup_cuda(params)

    batch_time = train_utils.AverageMeter('Time', ':5.3f')
    losses = train_utils.AverageMeter('Loss', ':.4e')
    top1 = train_utils.AverageMeter('Acc@1', ':5.2f')
    top5 = train_utils.AverageMeter('Acc@5', ':5.2f')
    progress = train_utils.ProgressMeter(len(params['val_dataloader']), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    params['model'].eval()

    end = time.time()
    for i, (data, target) in enumerate(params['val_dataloader']):

        # Pushes data to GPU
        data = data.to(params['device'])
        target = target.to(params['device'])

        # Generate adversarial attack (currently whitebox mode)
        if adversarial:
            
            data.requires_grad = True

            data = adversary.attack_batch(data, target, params['model'], 
                params['criterion'], attack_name='FGSM',
                device=params['device'], epsilon=0.3, alpha=0.5)
                
        with torch.no_grad():

            # compute output
            output = params['model'](data)
            loss = params['criterion'](output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % params['print_frequency'] == 0:
                progress.print(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # Storing validation losses/accuracies
    params['val_losses'].append(losses.get_avg())
    params['val_accuracies'].append(top1.get_avg())

    return top1.avg


# TODO: Allow this to be altered
def adjust_learning_rate(epoch, params):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = params['learning_rate'] * (0.33 ** (epoch // 5))
    for param_group in params['optimizer'].param_groups:
        param_group['lr'] = lr


# TODO: Test this and see if it works; if not, change it!
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def print_help():
    '''
    Lists all commands (currently) available in the
    main console.
    '''
    print('List of commands: ')
    print('-h: Help command. Prints this list of helpful commands!')
    print('-q: Quit. Immediately terminates the program.')
    print('-l: Load model. Loads a specific model/checkpoint into current program state.')
    print('-n: New model. Copies server metadata into local computer.')
    print('-p: Print. Prints the current program state (e.g. model, epoch, params, etc.)')
    print('-t: Train. Trains the network using the current program state.')
    print('-v: Validate. Runs the currently loaded network on the validation set.')
    print('-a: Adversarial. Runs the currently loaded network on adversarial examples.')
    print('-e: Edit. Gives the option to edit the current state.\n')


def main():
    params = param_factory()

    # Some error message about random seed
    if params['seed'] is not None:
        random.seed(params['seed'])
        torch.manual_seed(params['seed'])
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    while True:
        print('============== CS231n Project Console ==============')
        user_input = input('What would you like to do? (type -h for help) -> ')
        if user_input in ['-t', '--train', 't', 'train']:
            perform_training(params)
        elif user_input in ['-v', '--validate', 'v', 'validate']:
            validate(params)
        elif user_input in ['-l', '--load', 'l', 'load']:
            params = load_model(params)
        elif user_input in ['-n', '--new', 'n', 'new']:
            new_model(params)
        elif user_input in ['-h', '--help', 'h', 'help'] or user_input.strip() == '':
            print_help()
        elif user_input in ['-p', '--print', 'p', 'print']:
            print_state(params)
        elif user_input in ['-a', '--adversarial', 'a', 'adversarial']:
            validate(params, adversarial=True)
        elif user_input in ['-e', '--edit', 'e', 'edit']:
            edit_state(params)
        elif user_input in ['-q', '--quit', 'q', 'quit']:
            print()
            exit()
        else:
            print('Sorry, that command doesn\'t exist (yet)!')


if __name__ == '__main__':
    main()
