import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets


def get_mnist_dataloader(batch_sz=4, num_threads=1):

    """
    Downloads MNIST training and test sets into /datasets directory.
    
    Keyword arguments:
    > batch_sz -- Batch size to be grabbed from DataLoader
    > num_threads -- Number of threads with which to load data.

    Return value: (mnist_train_dataloader, mnist_test_dataloader)
    > mnist_train_dataloader -- a torch.utils.data.DataLoader wrapper around
        the MNIST training set.
    > mnist_test_dataloader -- a torch.utils.data.DataLoader wrapper around
        the MNIST test set.
    """

    # Downloads MNIST training and test sets into /datasets directory.
    mnist_train_set = datasets.MNIST(root='./datasets', train=True, download=True, transform=None)
    mnist_test_set = datasets.MNIST(root='./datasets', train=False, download=True, transform=None)

    # Constructs dataloader wrappers around MNIST training and test sets
    mnist_train_dataloader = DataLoader(mnist_train_set, batch_size=batch_sz, shuffle=True, num_workers=num_threads)
    mnist_test_dataloader = DataLoader(mnist_test_set, batch_size=batch_sz, shuffle=True, num_workers=num_threads)

    return mnist_train_dataloader, mnist_test_dataloader