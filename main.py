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
from train_utils import model_type, bolded # Ease of use

# For models
import models

# For constants
import constants

# For adversarial batch generation
import adversary

# For VAE-generated visualization
import viz_utils


def new_model(params):
    '''
    Creates a new model instance and initializes the respective fields
    in params.

    Keyword arguments:
    > params (dict) -- current state variable

    Returns: N/A
    '''

    print('You are initializing a new', bolded(model_type(params)), '.')
    model_list = constants.GENERATORS if params['is_generator'] else constants.CLASSIFIERS
    
    # Name
    params['run_name'] = input('Please type the current model run name -> ')

    # Architecture. Slightly hacky - allows constants.py to enforce 
    # which models are generators vs. classifiers.
    model_string = train_utils.input_from_list(model_list, 'model')
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
    elif model_string == 'VANILLA_VAE':
        params['model'] = models.VAE()
    else:
        raise Exception(model_string, 'does not exist as a model (yet)!')
        
    # Kaiming initialization for weights
    models.initialize_model(params['model'])

    # Setup other state variables
    for state_var in constants.SETUP_STATE_VARS:
        train_utils.store_user_choice(params, state_var)
        print()

    # Grabs dataloaders. TODO: Prompt for val split/randomize val indices
    params['train_dataloader'], params['val_dataloader'], params['test_dataloader'] = get_dataloader(
        dataset_name=params['dataset'], 
        batch_sz=params['batch_size'],
        num_threads=params['num_threads'])

    # Saves an initial copy
    if not os.path.isdir('models/' + model_type(params) + '/' + params['run_name'] + '/'):
        os.makedirs('models/' + model_type(params) + '/' + params['run_name'] + '/')
    train_utils.save_checkpoint(params, 0)


def load_model(params):
    '''
    Loads a model from a given checkpoint.
    '''
    
    # Grabs model path
    path = 'models/' + model_type(params) + '/'
    print('Attempting to load up a', bolded(model_type(params)), '.')
    
    # Grabs model directory from user
    model_folders = glob.glob(path + '*')
    if len(model_folders) == 0:
        print('No current models exist. Switching to creating a new model...')
        new_model(params)
        return

    # Grabs model choice from user
    print('\n --- All saved models ---')
#     model_folders = [model_folder[model_folder.rfind('/') + 1:] for model_folder in model_folders]
    user_model_choice = train_utils.input_from_list(model_folders, 'input')
    print('Chosen model:', user_model_choice[user_model_choice.rfind('/') + 1:])

    # Grabs checkpoint file from user
    print('\n --- All saved checkpoints ---')
    saved_checkpoint_files = glob.glob(user_model_choice + '/*')
#     saved_checkpoint_files = [checkpoint[checkpoint.rfind('/') + 1:] for checkpoint in saved_checkpoint_files]
    user_checkpoint_choice = train_utils.input_from_list(saved_checkpoint_files, 'checkpoint')
    print('Chosen checkpoint:', user_checkpoint_choice[user_checkpoint_choice.rfind('/') + 1:], '\n')

    # Loads saved state and sets up GPU for its model
    print('Loading model...')
    loaded = torch.load(user_checkpoint_choice)
    print('Finished loading model! Use -p to print current state.')
    
    # Backwards compatibility
    if 'is_generator' not in loaded:
        loaded['is_generator'] = False
        
    return loaded


def setup_cuda(params):
    '''
    Loads model onto GPU if one is available.
    '''
    print("Running on device: {}".format(params['device']))
    params['model'] = params['model'].to(params['device'])

    # Should make things faster if input size is consistent.
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
    cudnn.benchmark = True


def param_factory(is_generator=False):
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
    
    # Whether our model is a generator-type model
    params['is_generator'] = is_generator

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
    params['criterion'] = None

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
    print('Current state:', bolded(model_type(params)))
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
    

def perform_training(params):
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
    print('\n--- COMMENCE TRAINING ---\n')
    
    # Training/val loop
    for epoch in range(params['cur_epoch'] + 1, params['total_epochs'] + 1):
        
        print('Training: begin epoch', epoch)

        # LR Decay - currently a stepwise decay
        adjust_learning_rate(epoch, params)

        # train for one epoch
        train_one_epoch(epoch, params)

        # evaluate on validation set
        acc1 = validate(params, save=True, adversarial=False)

        # Update best val accuracy
        if not params['is_generator']:
            params['best_val_acc'] = max(acc1, params['best_val_acc'])

        # Save checkpoint every 'save_every' epochs.
        if epoch % params['save_every'] == 0:
            train_utils.save_checkpoint(params, epoch)

        # Update the current epoch
        params['cur_epoch'] += 1

        # Update train/val accuracy/loss plots
        viz_utils.plot_accuracies(params)
        viz_utils.plot_losses(params)

    train_utils.save_checkpoint(params, params['total_epochs'])
    print('\n--- END TRAINING ---\n')

    
def train_one_epoch(epoch, params):
    '''
    Trains model given in params['model'] for a single epoch.
    
    Keyword arguments:
    > epoch (int) -- current training epoch
    > params (dict) -- current state parameters
    
    Returns: N/A
    '''
    # Saves statistics about epoch (TODO -- pipe to file?)
    batch_time = train_utils.AverageMeter('Time', ':5.3f')
    data_time = train_utils.AverageMeter('Data', ':5.3f')
    losses = train_utils.AverageMeter('Loss', ':.4e')
    if params['is_generator']:
        progress = train_utils.ProgressMeter(len(params['train_dataloader']), batch_time, losses,
                                             prefix='Epoch: [{}]'.format(epoch))
    else:
        top1 = train_utils.AverageMeter('Acc@1', ':5.2f')
        top5 = train_utils.AverageMeter('Acc@5', ':5.2f')
        progress = train_utils.ProgressMeter(len(params['train_dataloader']), batch_time, losses,
                                             top1, top5, prefix='Epoch: [{}]'.format(epoch))

    # Switch to train mode. Important for dropout and batchnorm.
    params['model'].train()

    end = time.time()
    for i, (data, target) in enumerate(params['train_dataloader']):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Sends input/label tensors to GPU
        data = data.to(params['device'])
        target = target.to(params['device'])

        # Compute output
        output = params['model'](data)
        
        # Evaluate loss
        loss = params['criterion'](output, target)
        losses.update(loss.item(), data.size(0))

        # Measure accuracy and record loss
        if not params['is_generator']:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

        # Compute gradient and do SGD step
        params['optimizer'].zero_grad()
        loss.backward()
        params['optimizer'].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print training progress
        if i % params['print_frequency'] == 0:
            progress.print(i)
            
    # Prints final training accuracy per epoch
    progress.print(len(params['train_dataloader']))

    # Storing training losses/accuracies
    params['train_losses'].append(losses.get_avg())
    if not params['is_generator']:
        params['train_accuracies'].append(top1.get_avg())
    

def validate(params, save=False, adversarial=False, adversarial_attack=None, 
             whitebox=True, adversary_model=None, adversary_criterion=None):
    '''
    Performs validation of params['model'] using 
    params['val_dataloader']. 
    
    Keyword arguments:
    > params (dict) -- current state parameters
    > save (bool) -- whether to save val accuracies.
        Should only be `True` when called from train loop!
    > adversarial (bool) -- whether to test adversarially.
    > adversarial_attack (string) -- name of adversarial
        attack.
    > whitebox (bool) -- whether to use a whitebox attack.
    > adversary_model (torch.nn.Module) -- pre-trained
        model to generate black-box attacks with.
    > adversary_criterion (torch.nn.[loss]) -- loss func
        for the black-box "imitator" model.
        
    Returns: N/A
    '''
    if params['model'] is None:
        print('No model loaded! Type -n to create a new model, or -l to load an existing one from file.\n')
        return
    
    # Sets up training statistics to be logged to console (and possibly file -- TODO -- ?) output.
    if not adversarial:
        print('\n--- BEGIN VALIDATION PASS ---')
    batch_time = train_utils.AverageMeter('Time', ':5.3f')
    losses = train_utils.AverageMeter('Loss', ':.4e')
    if params['is_generator']:
        progress = train_utils.ProgressMeter(len(params['val_dataloader']), batch_time, losses,
                                             prefix='Test: ')
    else:
        top1 = train_utils.AverageMeter('Acc@1', ':5.2f')
        top5 = train_utils.AverageMeter('Acc@5', ':5.2f')
        progress = train_utils.ProgressMeter(len(params['val_dataloader']), batch_time, losses,
                                             top1, top5, prefix='Test: ')

    # Switch model to evaluate mode; push to GPU
    params['model'].eval()
    setup_cuda(params)

    end = time.time()
    for i, (data, target) in enumerate(params['val_dataloader']):

        # Pushes data to GPU
        data = data.to(params['device'])
        target = target.to(params['device'])

        # Generate adversarial attack (default whitebox mode)
        if adversarial:
            
            if whitebox:
                data = adversary.attack_batch(data, target, params['model'], 
                    params['criterion'], attack_name=adversarial_attack,
                    device=params['device'], epsilon=0.3, alpha=0.05)
            else:
                data = adversary.attack_batch(data, target, adversary_model, 
                    adversary_criterion, attack_name=adversarial_attack,
                    device=params['device'], epsilon=0.3, alpha=0.05)
                
        with torch.no_grad():

            # compute output
            output = params['model'](data)
            loss = params['criterion'](output, target)
            losses.update(loss.item(), data.size(0))
            
            # measure accuracy and record loss
            if not params['is_generator']:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % params['print_frequency'] == 0:
                progress.print(i)
    
    # Print final accuracy/loss
    progress.print(len(params['val_dataloader']))
    
    if not params['is_generator']:
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # Storing validation losses/accuracies
    if save:
        params['val_losses'].append(losses.get_avg())
        if not params['is_generator']:
            params['val_accuracies'].append(top1.get_avg())

    if not adversarial:
        print('--- END VALIDATION PASS ---\n')


def attack_validate(params):
    '''
    Performs all possible attacks (white-box only for now)
    in constants.ATTACKS.
    
    Keyword argument:
    params (dict) -- current state parameters
    
    Returns: N/A
    '''
    if train_utils.get_yes_or_no('Whitebox attack?'):
        for attack_name in constants.ATTACKS:
            print('Whitebox attack. Using current classifier (' + params['run_name'] + ').')
            print('\n--- COMMENCING ATTACK:', attack_name, '---')
            validate(params, save=False, adversarial=True, adversarial_attack=attack_name)
            print('--- ENDING ATTACK ---')
    else:
        print('\nBlackbox attack. Please load an imitator model.')
        imitator_state = load_model(param_factory(False))
        setup_cuda(imitator_state) # Sends imitator network to GPU
        for attack_name in constants.ATTACKS:
            print('\n--- COMMENCING ATTACK:', attack_name, '---')
            validate(params, save=False, adversarial=True, adversarial_attack=attack_name, 
                     whitebox=False, adversary_model=imitator_state['model'],
                     adversary_criterion=imitator_state['criterion'])
            print('--- ENDING ATTACK ---')
    print()

    
# TODO: Allow this to be altered
def adjust_learning_rate(epoch, params):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = params['learning_rate'] * (0.33 ** (epoch // 5))
    for param_group in params['optimizer'].param_groups:
        param_group['lr'] = lr


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

    
def defend(state_models):
    '''
    Constructs an attack/defense simulation using
    the currently loaded generator and classifier.
    
    Keyword arguments:
    > state_models (list of [dict, dict]) -- holds
        state parameters for classifier and generator,
        respectively.
        
    Returns: N/A
    '''
    # Constructs new defense state from generator/classifier states.
    classifier, generator = state_models[0]['model'], state_models[1]['model']
    defended_classifier = models.DAVAE(classifier, generator)
    defended_state = dict(state_models[0]) # Copies over state from classifier
    defended_state['model'] = defended_classifier
    defended_state['val_dataloader'] = state_models[1]['val_dataloader'] # Use generator dataloader
    
    if train_utils.get_yes_or_no('Adversarial test?'):
        if train_utils.get_yes_or_no('Whitebox attack?'):
            for attack_name in constants.ATTACKS:
                print('Whitebox attack. Using current model.')
                print('\n--- COMMENCING ATTACK:', attack_name, '---')
                validate(defended_state, save=False, adversarial=True, adversarial_attack=attack_name)
                print('--- ENDING ATTACK ---')
        else:
            print('\nBlackbox attack. Please load an imitator model.')
            imitator_state = load_model(param_factory(False))
            setup_cuda(imitator_state) # Sends imitator network to GPU
            for attack_name in constants.ATTACKS:
                print('\n--- COMMENCING ATTACK:', attack_name, '---')
                validate(defended_state, save=False, adversarial=True, adversarial_attack=attack_name, 
                         whitebox=False, adversary_model=imitator_state['model'],
                         adversary_criterion=imitator_state['criterion'])
                print('--- ENDING ATTACK ---')
        print()
    else:
        validate(defended_state, save=False, adversarial=False)
        
        

def print_models():
    '''
    Lists all currently saved models.
    '''
    model_folders = glob.glob('models/*')
    for folder in model_folders:
        print(folder[folder.rfind('/') + 1:])
        

def print_help(params, param_number):
    '''
    Lists all commands (currently) available in the
    main console.
    
    > Keyword arguments:
    params (dict) -- current state dict.
    param_number (int ~ [0, 1]) -- classifier or generator,
        respectively.
    '''
    print('List of commands: ')
    print('-h: Help command. Prints this list of helpful commands!')
    print('-q: Quit. Immediately terminates the program.')
    print('-l: Load model. Loads a specific model/checkpoint into current program state.')
    print('-n: New model. Creates new model from scratch.')
    print('-p: Print. Prints the current program state (e.g. model, epoch, params, etc.)')
    print('-t: Train. Trains the network using the current program state.')
    print('-v: Validate. Runs the currently loaded network on the validation set.')
    print('-e: Edit. Gives the option to edit the current state.')
    print('-m: Models. Lists all saved models.')
    print('-d: Defend. Runs the combined generator + classifier network on adversarial examples.')
    
    # Classifier state only
    if param_number == 0:
        print('-a: Adversarial. Runs the currently loaded network on adversarial examples.')
        
    # Generator state only
    elif param_number == 1:
        print('-g: Generate. Generate samples from the VAE.') # TODO: Make this more general
    else:
        raise Exception('Woah. How did you get here?')

    print()


def main():
    
    # Initialize state parameters
    # (classifier, generator)
    state_params = [param_factory(False), param_factory(True)]
    param_number = 0 # Classifier -> 0 || Generator -> 1
    
    # Create local directories
    train_utils.initialize_dirs()
    
#     # Some error message about random seed
#     if params['seed'] is not None:
#         random.seed(params['seed'])
#         torch.manual_seed(params['seed'])
#         cudnn.deterministic = True
#         warnings.warn('You have chosen to seed training. '
#                       'This will turn on the CUDNN deterministic setting, '
#                       'which can slow down your training considerably! '
#                       'You may see unexpected behavior when restarting '
#                       'from checkpoints.')

    # Console-style program
    while True:
        params = state_params[param_number]
        print('============== CS231n Project Console ==============')
        print('\nYou are currently in the', 
              bolded('classifier' if param_number == 0 else 'generator'), 'state.')
        print('Type `s` to swap states. Note that some commands are only available in one state.\n')
        user_input = input('What would you like to do? (type -h for help) -> ')
        if user_input in ['-t', '--train', 't', 'train']:
            perform_training(params)
        elif user_input in ['-v', '--validate', 'v', 'validate']:
            validate(params)
        elif user_input in ['-l', '--load', 'l', 'load']:
            state_params[param_number] = load_model(params)
        elif user_input in ['-n', '--new', 'n', 'new']:
            new_model(params)
        elif user_input in ['-h', '--help', 'h', 'help'] or user_input.strip() == '':
            print_help(params, param_number)
        elif user_input in ['-p', '--print', 'p', 'print']:
            print_state(params)
        elif user_input in ['-a', '--adversarial', 'a', 'adversarial']:
            if not params['is_generator']:
                attack_validate(params)
            else:
                print('Can\'t attack - model is not a classifier!')
                print('Use \'d\' to perform a defended attack.\n')
        elif user_input in ['-m', '--models', 'm', 'models']:
            print_models()
        elif user_input in ['-e', '--edit', 'e', 'edit']:
            edit_state(params)
        elif user_input in ['-g', '--generate', 'g', 'generate']:
            if params['is_generator']:
                viz_utils.sample_VAE(params['model'], params['device'],
                                     params['cur_epoch'], 'models/' + params['run_name'])
            else:
                print('Can\'t sample - model is not generative!')
                
        elif user_input in ['-d', '--defend', 'd', 'defend']:
            if state_params[1]['model'] is None:
                print('Can\'t defend - no generative model loaded.')
            elif state_params[0]['model'] is None:
                print('Can\'t defend - no classifier model loaded.')
            else:
                defend(state_params)
        elif user_input in ['-s', '--swap', 's', 'swap']:
            print('Switching from the', 
                  bolded('classifier' if param_number == 0 else 'generator'),
                  'state to the', bolded('classifier' if param_number == 1 else 'generator'),
                  'state.\n')
            
            # Yay group theory! Z / 2Z
            param_number = (param_number + 1) % 2
        elif user_input in ['-q', '--quit', 'q', 'quit']:
            print()
            exit()
        else:
            print('Sorry, that command doesn\'t exist (yet)!')


if __name__ == '__main__':
    main()
