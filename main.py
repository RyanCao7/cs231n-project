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

# For progress bars
import tqdm

# For DataLoader and transforms
from data_utils import get_dataloader

# For training utility functions
import train_utils

# For models
import models

# TODO: Add VAE and other architectures to this list
MODEL_ARCHITECTURE_NAMES = ['Classifier_A', 'Classifier_B', 'Classifier_C', 'Classifier_D', 'Classifier_E']
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256]
DATASETS = ['MNIST', 'CIFAR-10', 'FASHION-MNIST']
CRITERIA = ['CrossEntropyLoss'] # TODO: Add more criteria
OPTIMIZERS = ['SGD', 'Adam', 'RMSProp']


def new_model(params):
    '''
    Creates a new model instance and initializes the respective fields
    in params.

    Keyword arguments:
    > params (dict) -- current state variable

    Returns: N/A
    '''
    
    # Name
    params['run_name'] = input('Please type the current model run name -> ')

    # Architecture
    model_string = train_utils.input_from_list(MODEL_ARCHITECTURE_NAMES, 'model')
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
    models.initialize_model(params['model'])

    # Batch size
    params['batch_size'] = int(train_utils.input_from_list(BATCH_SIZES, 'batch size'))

    # Dataset
    params['dataset'] = train_utils.input_from_list(DATASETS, 'dataset')

    # Total epochs
    params['total_epochs'] = train_utils.input_from_range(1, 10000, 'training epochs')

    # Learning rate
    params['learning_rate'] = train_utils.input_float_range(0, 10, 'Learning rate')

    # Momentum
    params['momentum'] = train_utils.input_float_range(0, 1, 'Momentum')

    # Weight decay
    params['weight_decay'] = train_utils.input_float_range(0, 1, 'Weight decay')

    # Print frequency
    params['print_frequency'] = train_utils.input_from_range(1, 100, 'print frequency')

    # Default - checkpoint every 10 epochs
    params['save_every'] = train_utils.input_from_range(1, 100, 'save frequency')

    # Whether to evaluate on validation set
    params['evaluate'] = train_utils.get_yes_or_no('Evaluate on validation set?')

    # Random seed
    params['seed'] = train_utils.input_from_range(-1e99, 1e99, 'random seed')

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
    train_utils.save_checkpoint(params, epoch=0)


def load_model(params):
    '''
    Loads a model from a given checkpoint.
    '''
    # Grabs model directory from user
    model_folders = glob.glob('models/*')
    if len(model_folders) == 0:
        print('No current models exist. Switching to creating a new model...')
        new_model(params)

    user_model_choice = train_utils.input_from_list(model_folders, 'input')
    print('user model choice:', user_model_choice)

    # Grabs checkpoint file from user
    saved_checkpoint_files = glob.glob(user_model_choice + '/*')
    user_checkpoint_choice = train_utils.input_from_list(saved_checkpoint_files, 'checkpoint')

    # Loads saved state into params
    return torch.load(user_checkpoint_choice)


def param_factory():
    """
    Constructs a default parameter dictionary to be loaded up 
    upon start of console program.

    Keyword arguments: N/A

    Return value: params (dict)
    > params -- dictionary of default parameters.
    """
    params = {}

    # TODO: Fix defaults!
    params['dataset'] = 'MNIST'
    params['model'] = None
    params['load_workers'] = 4
    params['total_epochs'] = 90
    params['start_epoch'] = 0
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
    params['criterion'] = nn.CrossEntropyLoss().cuda(params['device'])

    # Optimizer - THIS MUST GET CONSTRUCTED LATER
    params['optimizer'] = None

    # Dataloaders
    params['train_dataloader'], params['test_dataloader'] = None, None

    return params

def print_help():
    print('List of commands: ')
    print('-h: Help command. Prints this list of helpful commands!')
    print('-q: Quit. Immediately terminates the program.')
    print('-l: Load model. Loads a specific model/checkpoint into current program state.')
    print('-n: New model. Copies server metadata into local computer.')
    print('-s: State. Prints the current program state (e.g. model, epoch, params, etc.)')
    print('-t: Train. Trains the network using the current program state.')
    print('-e: Evaluate. Evaluates the currently loaded network.')


# TODO: Actually print useful stuff :P
def print_state(params):
    print('Model:', params['model'], '\n')
    print('Epoch:', params['start_epoch'], '\n')
    print('Optimizer:', params['optimizer'], '\n')


def perform_training(params, evaluate=False):

    if params['model'] is None:
        print('You have no model! Please use -n to create a new model!')
        return

    # Load model onto GPU if one is available
    if params['device'] is not torch.device('cpu'):
        print("Use GPU: {} for training".format(params['device']))
        torch.cuda.set_device(params['device'])
        params['model'] = params['model'].cuda(params['device'])

    # Should make things faster if input size is consistent.
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
    cudnn.benchmark = True

    # Only test! TODO: Make this only test once. Have a separate `test(params)` function.`
    # if params['evaluate']:
    #     test(params)

    # Training/val loop
    for epoch in range(params['start_epoch'], params['epochs']):
        
        print('Training: begin epoch', epoch)

        # LR Decay - currently a stepwise decay
        adjust_learning_rate(epoch, params)

        # train for one epoch
        train(params)

        # evaluate on validation set
        acc1 = validate(params)

        # Update best val accuracy
        params['best_val_acc'] = max(acc1, params['best_val_acc'])

        # Update the starting epoch
        params['start_epoch'] += 1

        # Save checkpoint every 'save_every' epochs.
        if epoch % params['save_every'] == 0:
            train_utils.save_checkpoint(params, epoch)


# From PyTorch. TODO: Remove (but not yet)
def load_checkpoint(params):
    # optionally resume from a checkpoint
    if params.resume:
        if os.path.isfile(params.resume):
            print("=> loading checkpoint '{}'".format(params.resume))
            checkpoint = torch.load(params.resume)
            params['start_epoch'] = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if params.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(params.gpu)
            params['model'].load_state_dict(checkpoint['state_dict'])
            params['optimizer'].load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(params.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(params.resume))


def train(params):

    # No idea what this is for now...
    batch_time = train_utils.AverageMeter('Time', ':6.3f')
    data_time = train_utils.AverageMeter('Data', ':6.3f')
    losses = train_utils.AverageMeter('Loss', ':.4e')
    top1 = train_utils.AverageMeter('Acc@1', ':6.2f')
    top5 = train_utils.AverageMeter('Acc@5', ':6.2f')
    progress = train_utils.ProgressMeter(len(params['train_dataloader']), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(params['epoch']))

    # Switch to train mode. Important for dropout and batchnorm.
    params['model'].train()

    end = time.time()
    for i, (input, target) in enumerate(params['train_loader']):
        # measure data loading time
        data_time.update(time.time() - end)

        if params.gpu is not None:
            input = input.cuda(params.gpu, non_blocking=True)
        target = target.cuda(params.gpu, non_blocking=True)

        # compute output
        output = params['model'](input)
        loss = params['criterion'](output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        params['optimizer'].zero_grad()
        loss.backward()
        params['optimizer'].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print training progress
        if i % params['print_freq'] == 0:
            progress.print(i)


def validate(params):
    batch_time = train_utils.AverageMeter('Time', ':6.3f')
    losses = train_utils.AverageMeter('Loss', ':.4e')
    top1 = train_utils.AverageMeter('Acc@1', ':6.2f')
    top5 = train_utils.AverageMeter('Acc@5', ':6.2f')
    progress = train_utils.ProgressMeter(len(params['val_dataloader']), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    params['model'].eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(params['val_dataloader']):
            if params.gpu is not None:
                input = input.cuda(params.gpu, non_blocking=True)
            target = target.cuda(params.gpu, non_blocking=True)

            # compute output
            output = params['model'](input)
            loss = params['criterion'](output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % params['print_freq'] == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


# TODO: Allow this to be altered
def adjust_learning_rate(epoch, params):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = params['learning_rate'] * (0.1 ** (epoch // 30))
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
        user_input = input('What would you like to do? (type -h for help) -> ')
        if user_input in ['-t', '--train', 't', 'train']:
            perform_training(params)
        elif user_input in ['-e', '--eval', 'e', 'eval']:
            perform_training(params, evaluate=True)
        elif user_input in ['-l', '--load', 'l', 'load']:
            params = load_model(params)
        elif user_input in ['-n', '--new', 'n', 'new']:
            new_model(params)
        elif user_input in ['-h', '--help', 'h', 'help']:
            print_help()
        elif user_input in ['-s', '--state', 's', 'state']:
            print_state(params)
        elif user_input in ['-q', '--quit', 'q', 'quit']:
            exit()
        else:
            print('Sorry, that command doesn\'t exist (yet)!')


if __name__ == '__main__':
    main()