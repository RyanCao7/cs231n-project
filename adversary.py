import torch
import torch.nn as nn
import torch.nn.functional as F


def attack_batch(batch, target, model, loss_fcn, attack_name='FGSM', 
                 device=torch.device('cpu'), epsilon=0.3, alpha=0.5):
    """
    Generated adversarial versions of a batch for a given attack.
    
    Keyword arguments:
    > batch (tensor) -- Collection of images to attack.
    > target (tensor) -- Target classes for each image in input batch.
    > model (nn.Module) -- Model with respect to the attack is being done.
    > loss_fcn (function) -- Loss function for use with 'model'. Must use
        interface (output, target).
    > attack_name (string) -- Name of the attack to use. Must be one of
        {'FGSM', 'RAND-FGSM', 'CW'}.
    > device (torch.device) -- Device for model computation.
    > epsilon (float) -- Weight for data gradient used in FGSM. Default value
        taken from Defense GAN paper.
    > alpha (float) -- Weight for gaussian noise in RAND-FGSM. Default value 
        taken from Defense GAN paper.
    
    Return value: adversarial_batch
    > adversarial_batch (tensor) -- adversarial version of input batch.
    """
    if attack_name == 'FGSM':
        return fgsm_attack(batch, epsilon, get_batch_grad(batch, target, model, 
                                                          loss_fcn, device))
    elif attack_name == 'RAND_FGSM':
        noisy_batch = batch + alpha * torch.rand_like(batch)
        return fgsm_attack(noisy_batch, epsilon, get_batch_grad(noisy_batch,
                                                                target, model,
                                                                loss_fcn, device))
    elif attack_name == 'CW':
        # TODO Implement Carlini-Wagner Attack
        pass
    else:
        raise Exception('Error: attack_name must be one of {\'FGSM\', \'RAND_FGSM\', '
                        '\'CW\'}.')

    
def fgsm_attack(batch, epsilon, batch_grad):
    """
    Performs the Fast Gradient Sign Method (FGSM) attack as described by
    Goodfellow et al. in "Explaining and Harnessing Adversarial Examples".

    Keyword arguments:
    > batch (tensor) -- Collection of images to attack.
    > epsilon (float) -- Weight for data gradient used.
    > batch_grad (tensor) -- Loss gradient with respect to batch.

    Return value: perturbed_batch
    > perturbed_batch (tensor) -- version of input batch perturbed by data
        gradient.
    """
    sign_batch_grad = batch_grad.sign()
    perturbed_batch = batch + epsilon * sign_batch_grad
    perturbed_batch = torch.clamp(perturbed_batch, 0, 1)
    return perturbed_batch

def get_batch_grad(batch, target, model, loss_fcn, device):
    """
    Generated adversarial versions of a batch for a given attack.
    
    Keyword arguments:
    > batch (tensor) -- Collection of images to attack.
    > target (tensor) -- Target classes for each image in input batch.
    > model (nn.Module) -- Model with respect to the attack is being done.
    > loss_fcn (function) -- Loss function for use with 'model'. Must use
        interface (output, target).
    > device (torch.device) -- Device for model computation.
    
    Return value: batch_grad
    > batch_grad (tensor) -- Loss gradient with respect to batch.
    """
    
    batch = batch.to(device)
    target = target.to(device)

    model.eval()
    batch.requires_grad = True

    output = model(batch)
    loss = loss_fcn(output, target)

    model.zero_grad()
    loss.backward()
    return batch.grad.data
    
