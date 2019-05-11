import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets


def get_dataloader(dataset_name='MNIST', batch_sz=4, num_threads=1):

    """
    Downloads specified dataset's training and test sets into /datasets directory.
    
    Keyword arguments:
    > dataset_name -- Name of dataset to be loaded. Must be one of
        {'MNIST', 'CIFAR-10'}.
    > batch_sz -- Batch size to be grabbed from DataLoader.
    > num_threads -- Number of threads with which to load data.

    Return value: (train_dataloader, test_dataloader)
    > train_dataloader -- a torch.utils.data.DataLoader wrapper around
        the specified dataset's training set.
    > test_dataloader -- a torch.utils.data.DataLoader wrapper around
        the specified dataset's test set.
    """

    train_set, test_set = None, None

    # Downloads requested training and test sets into /datasets directory.
    if dataset_name == 'MNIST':
        train_set = datasets.MNIST(root='./datasets', train=True, download=True, transform=None)
        test_set = datasets.MNIST(root='./datasets', train=False, download=True, transform=None)
    elif dataset_name == 'CIFAR-10':
        train_set = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=None)
        test_set = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=None)
    else:
        raise Exception('Error: dataset_name must be one of {\'MNIST\', \'CIFAR-10\'}.')

    # Constructs dataloader wrappers around MNIST training and test sets
    train_dataloader = DataLoader(train_set, batch_size=batch_sz, shuffle=True, num_workers=num_threads)
    test_dataloader = DataLoader(test_set, batch_size=batch_sz, shuffle=True, num_workers=num_threads)

    return train_dataloader, test_dataloader