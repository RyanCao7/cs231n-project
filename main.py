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
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# For progress bars
import tqdm

# For DataLoader and transforms
from data_utils import get_dataloader

# For training utility functions
from train_utils import AverageMeter, ProgressMeter, save_checkpoint


# HUGE TODO: Prompt and set parameters for a new model, and
# load that state into params
def new_model(params):
    pass


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
    params['all_architectures'] = [] # Fix this!
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
    """torch.optim.SGD(model.parameters(), params['lr'],
                                momentum=params['momentum'],
                                weight_decay=params['weight_decay'])"""
    return params


def main():
    params = param_factory()

    if params['seed'] is not None:
        random.seed(params['seed'])
        torch.manual_seed(params['seed'])
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Simply call main_worker function
    main_worker(params)


def main_worker(params):
    global best_acc1

    # TODO: Fix where the model loads from
    # Load model here...
    print("=> creating model '{}'".format(params['model']))

    # Load model onto GPU if one is available
    if params['device'] is not None:
        print("Use GPU: {} for training".format(params['device']))
        torch.cuda.set_device(params['device'])
        params['model'] = params['model'].cuda(params['device'])

    # TODO: Look this up
    cudnn.benchmark = True

    # Data loading code
    train_dataloader, test_dataloader = get_dataloader(dataset_name=params['dataset'], 
        batch_sz=params['batch_size'], num_threads=params['num_threads'])

    # Evaluate once at the beginning (sanity check)
    if params['evaluate']:
        validate(params)
        return

    # Training/val loop
    for epoch in range(params['start_epoch'], params['epochs']):

        # LR Decay (TODO) - look into this
        adjust_learning_rate(params)

        # train for one epoch
        train(params)

        # evaluate on validation set
        acc1 = validate(params)

        # Update best val accuracy
        params['best_val_acc'] = max(acc1, params['best_val_acc'])

        # Save checkpoint every 'save_every' epochs.
        if epoch % params['save_every'] == params['save_every']:
            save_checkpoint(params, epoch)


def load_checkpoint(params):
    # List all models, and all epochs
    for folder_name in 

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

def train(train_loader, model, criterion, optimizer, epoch, params):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if params.gpu is not None:
            input = input.cuda(params.gpu, non_blocking=True)
        target = target.cuda(params.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % params.print_freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion, params):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if params.gpu is not None:
                input = input.cuda(params.gpu, non_blocking=True)
            target = target.cuda(params.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % params.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def adjust_learning_rate(optimizer, epoch, params):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = params.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
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


if __name__ == '__main__':
    main()