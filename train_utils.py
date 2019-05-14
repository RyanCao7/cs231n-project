import torch

# TODO: Make this save for best val accuracy epoch
DEFAULT_SAVE_EXTENSION = '.pth.tar'
def save_checkpoint(params, epoch):
    torch.save(params, 'models/' + params['run_name'] + '/' + params['run_name'] + '_epoch_' + str(epoch) + DEFAULT_SAVE_EXTENSION)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')


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
    input_idx = int(input('Please type the number of the ' + item + ' you wish to be loaded (enter for default) -> ')) - 1
    while input_idx not in range(len(the_list)):
        input_idx = int(input('Try again. Please type the number of the ' + item + ' you wish to be loaded (enter for default) -> ')) - 1
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