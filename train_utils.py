import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import glob
import constants
import loss_functions
import box_utils
import viz_utils
from adversary import attack_batch


# TODO: Make this save for best val accuracy epoch
DEFAULT_SAVE_EXTENSION = '.pth.tar'
def save_checkpoint(params, epoch):
    '''
    Saves the parameter dictionary passed in.
    `epoch` is assumed to be the number of training 
    epochs completed by the model (i.e. begin 
    training at epoch + 1).

    Keyword arguments:
    > params (dict) -- current state variable
    > epoch (int) -- number of COMPLETED training epochs

    Returns: N/A
    '''
    save_path = 'models/' + model_type(params) + '/' + params['run_name'] \
                + '/' + params['run_name'] + '_epoch_' + str(epoch) + \
                '~' + constants.get_cur_time() + DEFAULT_SAVE_EXTENSION
    
    print('Saving current state to', save_path)
    torch.save(params, save_path)
    
    # Uploads single saved checkpoint to Box
    viz_utils.rid_duplicate(save_path)
    box_utils.upload_single(save_path)

    
def sample_from_dataset(params):
    '''
    Because DataLoader API doesn't allow us to easily/
    cleanly sample randomly from it...
    '''
    random_idx = int(np.random.random() * len(params['val_dataloader']))
    for i, (data, target) in enumerate(params['val_dataloader']):
        if i == random_idx:
            return data


def sample_attack_from_dataset(params, attack_name='FGSM', epsilon=0.3, alpha=0.05, data=None):
    '''
    Because DataLoader API doesn't allow us to eaily/
    cleanly sample randomly from it...
    '''
    if data == None:
        random_idx = int(np.random.random() * len(params['val_dataloader']))
        for i, (data, target) in enumerate(params['val_dataloader']):
            if i == random_idx:
                data = data.to(params['device'])
    return data, attack_batch(data, target, params['model'], params['criterion'], 
                        attack_name=attack_name, device=params['device'], 
                        epsilon=epsilon, alpha=alpha)


def model_type(params):
    '''
    One-liner helper function for printing
    correct model type.
    '''
    return ('generator' if params['is_generator'] else 'classifier')
    

def initialize_dirs():
    '''
    Creates local needed directories for saving data.
    '''
    if not os.path.isdir('datasets/'):
        os.makedirs('datasets/')
    if not os.path.isdir('models/classifier/'):
        os.makedirs('models/classifier/')
    if not os.path.isdir('models/generator/'):
        os.makedirs('models/generator/')
    if not os.path.isdir('graphs/'):
        os.makedirs('graphs/')
    if not os.path.isdir('visuals/'):
        os.makedirs('visuals/')
    

class color:
    '''
    Color constants for colored text.
    '''
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def bolded(msg):
    '''
    Bolds the given string when printed.
    '''
    return color.BOLD + msg + color.END


def store_user_choice(params, keyword):
    '''
    Utility function for editing a single parameter in params
    given by keyword.

    Keyword arguments:
    > params (dict) -- currently loaded state dict.
    > keyword (string) -- key for state variable in params to
        be edited.

    Returns: N/A
    '''
    if keyword == 'batch_size':
        params['batch_size'] = int(input_from_list(constants.BATCH_SIZES, 'batch size'))
    elif keyword == 'dataset':
        params['dataset'] = input_from_list(constants.DATASETS, 'dataset')
    elif keyword == 'total_epochs':
        params['total_epochs'] = input_from_range(1, 10000, 'training epochs')
    elif keyword == 'learning_rate':
        params['learning_rate'] = input_float_range(0, 10, 'Learning rate')
    elif keyword == 'momentum':
        params['momentum'] = input_float_range(0, 1, 'Momentum')
    elif keyword == 'weight_decay':
        params['weight_decay'] = input_float_range(0, 1, 'Weight decay')
    elif keyword == 'print_frequency':
        params['print_frequency'] = input_from_range(1, 10000, 'print frequency')
    elif keyword == 'save_every':
        params['save_every'] = input_from_range(1, 100, 'save frequency')
    elif keyword == 'evaluate':
        params['evaluate'] = get_yes_or_no('Evaluate on validation set?')
    elif keyword == 'seed':
        params['seed'] = input_from_range(-1e99, 1e99, 'random seed')
    elif keyword == 'adversarial_train':
        params['adversarial_train'] = get_yes_or_no('Train with adversarial training?')
    elif keyword == 'alpha':
        params['alpha'] = input_float_range(0, 1, 'Desired alpha value (for adversarial training loss weighting)')
        
    # Optimizer choice
    elif keyword == 'optimizer':
        optimizer_choice = input_from_list(constants.OPTIMIZERS, 'optimizer')
        if optimizer_choice == 'SGD':
            params['optimizer'] = torch.optim.SGD(params['model'].parameters(),
                                                  params['learning_rate'],
                                                  momentum=params['momentum'],
                                                  weight_decay=params['weight_decay'])
        elif optimizer_choice == 'Adam':
            params['optimizer'] = torch.optim.Adam(params['model'].parameters(),
                                                  params['learning_rate'],
                                                  betas=(0.9, 0.999), # TODO: ALLOW USER TO CHOOSE ADAM BETAS
                                                  weight_decay=params['weight_decay'])
        else:
            raise Exception('Optimizer', optimizer_choice, ' does not exist (yet).')
            
    # Loss function
    elif keyword == 'criterion':
        criteria = (constants.GENERATOR_CRITERIA if params['is_generator'] 
                    else constants.CLASSIFIER_CRITERIA)
        criterion_choice = input_from_list(criteria, 'criterion')
        if criterion_choice == 'CrossEntropy':
            params['criterion'] = nn.CrossEntropyLoss().to(params['device'])
        elif criterion_choice == 'ReconKLD':
            params['criterion'] = loss_functions.ReconKLD().to(params['device'])
        else:
            raise Exception('Criterion', criterion_choice, ' does not exist (yet).')
    else:
        print('\'' + keyword + '\'', 'is not editable in state dict.')

        
def float_input(prompt):
    '''
    Robust float input getter.
    '''
    while True:
        try:
            user_input = float(input(prompt))
            return user_input
        except ValueError:
            print('Please enter a float!')
            pass

    
def int_input(prompt):
    '''
    Robust int input getter.
    '''
    while True:
        try:
            user_input = int(input(prompt))
            return user_input
        except ValueError:
            print('Please enter an int!')
            pass

        
# TODO: Implement default
def input_float_range(low, high, prompt):
    '''
    Utility helper function to grab a user's choice
    of integer from an inclusive range.

    Keyword Arguments: low (float), high (float), prompt (string)
    > low -- lower bound
    > high -- upper bound
    > prompt -- message to prompt with

    Returns: user_input (float)
    > user_input -- user's final input, within [low, high].
    '''
    user_input = float_input(prompt + '? (range: [' + str(low) + ', ' + str(high) + ']) -> ')
    if user_input < low or user_input > high:
        print('Error: must be within ' + str(low) + ' to ' + str(high) + ', inclusive.')
        user_input = float_input(prompt + '? (range: [' + str(low) + ', ' + str(high) + ']) -> ')
    return user_input


def get_yes_or_no(prompt):
    '''
    Simply returns a bool based on user yes/no.
    '''
    answer = input(prompt + ' (y/n) -> ')
    while answer.lower() not in ['y', 'yes', 'n', 'no']:
        print('Please enter yes/no.')
        answer = input(prompt + ' (y/n) -> ')
    return answer.lower() in ['y', 'yes']


# TODO: Implement default
def input_from_range(low, high, prompt):
    '''
    Utility helper function to grab a user's choice
    of integer from an inclusive range.

    Keyword Arguments: low (int), high (int), prompt (string)
    > low -- lower bound
    > high -- upper bound
    > prompt -- message to prompt with

    Returns: user_input (int)
    > user_input -- user's final input, within [low, high].
    '''
    user_input = int_input('Number of ' + prompt + '? (range: [' + str(low) + ', ' + str(high) + ']) -> ')
    if user_input < low or user_input > high:
        print('Error: must be within ' + str(low) + ' to ' + str(high) + ', inclusive.')
        user_input = int_input('Number of ' + prompt + '? (range: [' + str(low) + ', ' + str(high) + ']) -> ')
    return user_input


# TODO: Implement default
def input_from_list(the_list, item, default=None):
    '''
    Utility helper function to grab a user's choice
    of object from a list.

    Keyword Arguments: the_list (list<T>), item (string)
    > the_list -- a list of items to choose from.
    > item -- the type of item to be stated in the prompt.

    Returns: the_list[input_idx] (T)
    > the_list[input_idx] -- the `input_idx`th object in the list,
        as specified by the user.
    '''

    the_list = sorted(the_list)
    for idx, list_item in enumerate(the_list):
        print(str((idx + 1)) + ':', list_item)
    input_idx = int_input('Please type the index of the ' + item + ' you wish to choose (enter for default) -> ') - 1
    while input_idx not in range(len(the_list)):
        input_idx = int_input('Try again. Please type the index of the ' + item + ' you wish to choose (enter for default) -> ') - 1
    return the_list[input_idx]


def delete_future_checkpoints(params):
    '''
    Deletes all old future epoch training checkpoints for consistency
    (i.e. all saved weights are continuums of the same training process).
    
    Keyword Arguments: params (dict)
    > params -- current parameter state dictionary
    '''
    path = 'models/' + model_type(params) + '/'
    print('Warning: training will delete all checkpoints past the current (' +
                         str(params['cur_epoch'] + 1) + ') epoch:')
    to_be_deleted = []
    for checkpoint in glob.glob(path + params['run_name'] + '/*.pth.tar'):
        run_num = int(checkpoint[checkpoint.rfind('_') + 1 : checkpoint.find('~')])
        if run_num > params['cur_epoch'] + 1:
            print('To be deleted:', checkpoint)
            to_be_deleted.append(checkpoint)
    if not get_yes_or_no('Continue?'):
        return False
    
    print('Deleting all checkpoints past ' + str(params['cur_epoch'] + 1) + ' epoch...')
    for checkpoint in to_be_deleted:
        print('deleting', checkpoint + '...')
        status = subprocess.call(['rm', checkpoint])
        assert(status == 0)
    return True


# Taken from PyTorch/examples
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def get_avg(self):
        return self.avg


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
