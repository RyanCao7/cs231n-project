import torch
import constants
import matplotlib.pyplot as plt


# TODO: Make this save for best val accuracy epoch
DEFAULT_SAVE_EXTENSION = '.pth.tar'
def save_checkpoint(params, epoch):
    print('Saving current state to', 
        'models/' + params['run_name'] + '/' + params['run_name'] + '_epoch_' + str(epoch) + DEFAULT_SAVE_EXTENSION)
    torch.save(params, 'models/' + params['run_name'] + '/' + params['run_name'] + '_epoch_' + str(epoch) + DEFAULT_SAVE_EXTENSION)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')


def plot_accuracies(params):
    plt.figure(figsize=(10, 8))
    plt.title('Accuracies')
    plt.plot(params['train_accuracies'], '-o', label='Training Accuracy')
    plt.plot(params['val_accuracies'], '-o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    # plt.gcf().set_size_inches(15, 12)
    plt.savefig('models/' + params['run_name'] + '/' + params['run_name'] + '_accuracies.png')
    plt.close()
 

def plot_losses(params):
    plt.figure(figsize=(10, 8))
    plt.title('Losses')
    plt.plot(params['train_losses'], '-o', label='Training Loss')
    plt.plot(params['val_losses'], '-o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    # plt.gcf().set_size_inches(15, 12)
    plt.savefig('models/' + params['run_name'] + '/' + params['run_name'] + '_losses.png')
    plt.close()


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
        params['print_frequency'] = input_from_range(1, 100, 'print frequency')
    elif keyword == 'save_every':
        params['save_every'] = input_from_range(1, 100, 'save frequency')
    elif keyword == 'evaluate':
        params['evaluate'] = get_yes_or_no('Evaluate on validation set?')
    elif keyword == 'seed':
        params['seed'] = input_from_range(-1e99, 1e99, 'random seed')
    else:
        print('\'' + keyword + '\'', 'is not editable in state dict.')

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
    user_input = float(input(prompt + '? (range: [' + str(low) + ', ' + str(high) + ']) -> '))
    if user_input < low or user_input > high:
        print('Error: must be within ' + str(low) + ' to ' + str(high) + ', inclusive.')
        user_input = float(input(prompt + '? (range: [' + str(low) + ', ' + str(high) + ']) -> '))
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
    user_input = int(input('Number of ' + prompt + '? (range: [' + str(low) + ', ' + str(high) + ']) -> '))
    if user_input < low or user_input > high:
        print('Error: must be within ' + str(low) + ' to ' + str(high) + ', inclusive.')
        user_input = int(input('Number of ' + prompt + '? (range: [' + str(low) + ', ' + str(high) + ']) -> '))
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

    for idx, list_item in enumerate(the_list):
        print(str((idx + 1)) + ':', list_item)
    input_idx = int(input('Please type the index of the ' + item + ' you wish to choose (enter for default) -> ')) - 1
    while input_idx not in range(len(the_list)):
        input_idx = int(input('Try again. Please type the index of the ' + item + ' you wish to choose (enter for default) -> ')) - 1
    return the_list[input_idx]


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