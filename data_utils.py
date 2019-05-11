import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets


def get_dataloader(dataset_name='MNIST', batch_sz=4, num_threads=1):

    """
    Downloads specified dataset's training and test sets into /datasets directory.
    
    Keyword arguments:
    > dataset_name -- Name of dataset to be loaded. Must be one of
        {'MNIST', 'CIFAR-10'}
    > batch_sz -- Batch size to be grabbed from DataLoader
    > num_threads -- Number of threads with which to load data.

    Return value: (mnist_train_dataloader, mnist_test_dataloader)
    > mnist_train_dataloader -- a torch.utils.data.DataLoader wrapper around
        the MNIST training set.
    > mnist_test_dataloader -- a torch.utils.data.DataLoader wrapper around
        the MNIST test set.
    """

    train_set, test_set = None, None

    if dataset_name == 'MNIST':
        # Downloads MNIST training and test sets into /datasets directory.
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