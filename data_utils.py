import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch.utils.data.sampler import SubsetRandomSampler


def get_dataloader(dataset_name='MNIST', val_split=0.2, batch_sz=4, num_threads=1, shuffle_val=True):
    """
    Downloads specified dataset's training and test sets into /datasets directory.
    
    Keyword arguments:
    > dataset_name (string) -- Name of dataset to be loaded. Must be one of
        {'MNIST', 'CIFAR-10', 'Fashion-MNIST'}.
    > val_split (float) -- Fraction of training data to be used as validation set.
    > batch_sz (int) -- Batch size to be grabbed from DataLoader.
    > num_threads (int) -- Number of threads with which to load data.
    > shuffle_val (bool) -- Whether to shuffle validation set indices.

    Return value: (train_dataloader, val_dataloader, test_dataloader)
    > train_dataloader -- a torch.utils.data.DataLoader wrapper around
        the specified dataset's training set.
    > val_dataloader -- a torch.utils.data.DataLoader wrapper around
        the specified dataset's validation set.
    > test_dataloader -- a torch.utils.data.DataLoader wrapper around
        the specified dataset's test set.
    """

    # Declares train/test sets and sets up transforms
    train_set, test_set = None, None
    data_transforms = transform_factory(dataset_name=dataset_name)
    train_transform, test_transform = data_transforms['train'], data_transforms['test']

    # Downloads requested training and test sets into /datasets directory.
    if dataset_name == 'MNIST':
        train_set = datasets.MNIST(root='./datasets', train=True, download=True, 
                                   transform=train_transform)
        test_set = datasets.MNIST(root='./datasets', train=False, download=True, 
                                  transform=test_transform)
    elif dataset_name == 'CIFAR-10':
        train_set = datasets.CIFAR10(root='./datasets', train=True, download=True, 
                                     transform=train_transform)
        test_set = datasets.CIFAR10(root='./datasets', train=False, download=True, 
                                    transform=test_transform)
    elif dataset_name == 'Fashion-MNIST':
        train_set = datasets.FashionMNIST(root='./datasets', train=True, download=True, 
                                          transform=train_transform)
        test_set = datasets.FashionMNIST(root='./datasets', train=False, download=True, 
                                         transform=test_transform)
    else:
        raise Exception('Error: dataset_name must be one of {\'MNIST\', \'CIFAR-10\', '
                        '\'Fashion-MINST\'}.')

    # Grabs train/val split
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    if shuffle_val:
        # np.random.seed(random_seed) # We may need this...?
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    # Constructs dataloader wrappers around MNIST training and test sets
    train_dataloader = DataLoader(train_set, batch_size=batch_sz, 
                                  num_workers=num_threads, sampler=train_sampler)
    val_dataloader = DataLoader(train_set, batch_size=batch_sz, 
                                num_workers=num_threads, sampler=val_sampler)
    test_dataloader = DataLoader(test_set, batch_size=batch_sz, 
                                 shuffle=True, num_workers=num_threads)

    return train_dataloader, val_dataloader, test_dataloader


def transform_factory(dataset_name='MNIST'):
    """
    Constructs corresponding transform dictionary according to
    dataset_name and train/test flag. Currently only performs
    toTensor() and Normalize() transforms.

    Keyword arguments:
    > dataset_name -- Name of dataset to be transformed. Must be one of
        {'MNIST', 'CIFAR-10', 'Fashion-MNIST'}.

    Return value: data_transforms
    > data_transforms -- a dictionary that can be passed into a DataLoader
        to process/augment an MNIST/CIFAR-10 dataset.
    """
    
    # MNIST mean and std pulled from 
    # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
    # CIFAR-10 mean and std pulled from 
    # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151 (see first comment)
    # with reference to
    # https://github.com/tomgoldstein/loss-landscape/blob/master/cifar10/dataloader.py#L16
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)
    FASHIONMNIST_MEAN = (0.2860407,)
    FASHIONMNIST_STD = (0.35302424,)

    normalize_transform = None
    if dataset_name == 'MNIST':
        normalize_transform = transforms.Normalize(MNIST_MEAN, MNIST_STD)
    elif dataset_name == 'CIFAR-10':
        normalize_transform = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    elif dataset_name == 'Fashion-MNIST':
        normalize_transform = transforms.Normalize(FASHIONMNIST_MEAN, FASHIONMNIST_STD)
    else:
        raise Exception('Error: dataset_name must be one of {\'MNIST\', \'CIFAR-10\', '
                        '\'Fashion-MINST\'}.')

    # TODO: Write meaningful transforms for train/val/test sets.
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
#             normalize_transform
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
#             normalize_transform
        ]),
    }

    return data_transforms


# Example of how to iterate through dataloader.
if __name__ == '__main__':
    train_dataloader, _, _ = get_dataloader('Fashion-MNIST', val_split=0, batch_sz=60000)
    print(train_dataloader)
    for idx, thing in enumerate(train_dataloader):
        if idx == 0:
            examples, labels = thing
            print(examples.shape) # Should be (N, C, H, W)
            print(labels.shape) # Should be (N,)

            print(np.mean(examples.numpy())) # Should be close to 0
            print(np.std(examples.numpy())) # Should be close to 1
            break