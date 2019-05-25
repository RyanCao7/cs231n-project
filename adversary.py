import torch
import torch.nn as nn
import torch.nn.functional as F


def attack_batch(batch, target, model, loss_fcn, attack_name='FGSM', 
                 device=torch.device('cpu'), epsilon=0.3, alpha=0.05,
                 lr = 10.0, num_iter=100, c=100.0, min=0, max=1):
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
    > lr (float) -- Learning rate for CW projected GD. Default value taken
        from Defense GAN paper.
    > num_iter (int) -- Number of iterations to perform PGD. Default value
        taken from Defense GAN paper.
    > c (float) -- Weight constant for CW objective function. Default value
        taken from Defense GAN paper
    > min (float) -- Minimum value that can occur in an image. Default value
        assumes no normalization has taken place.
    > max (float) -- Maximum value that can occur in an image. Default value
        assumes no normalization has taken place.

    Return value: adversarial_batch
    > adversarial_batch (tensor) -- adversarial version of input batch.
    """

    batch = batch.to(device)
    target = target.to(device)
    model.eval()

    if attack_name == 'FGSM':
        return fgsm_attack(batch, epsilon, get_batch_grad(batch, target, model, 
                                                          loss_fcn, device),
                           min, max)
    elif attack_name == 'RAND_FGSM':
        noisy_batch = batch + alpha * torch.sign(torch.randn_like(batch))
        return fgsm_attack(noisy_batch, epsilon - alpha, get_batch_grad(noisy_batch,
                                                                target, model,
                                                                loss_fcn, device),
                           min, max)
    elif attack_name == 'CW':
        return cw_attack(batch, target, model, device, lr, num_iter, c, min, max)
    else:
        raise Exception('Error: attack_name must be one of {\'FGSM\', \'RAND_FGSM\', '
                        '\'CW\'}.')

    
def fgsm_attack(batch, epsilon, batch_grad, min, max):
    """
    Performs the Fast Gradient Sign Method (FGSM) attack as described by
    Goodfellow et al. in "Explaining and Harnessing Adversarial Examples".

    Keyword arguments:
    > batch (tensor) -- Collection of images to attack.
    > epsilon (float) -- Weight for data gradient used.
    > batch_grad (tensor) -- Loss gradient with respect to batch.
    > max (float) -- The maximum value possible in a valid image. 
    > min (float) -- The minimum value possible in a valid image. 

    Return value: perturbed_batch
    > perturbed_batch (tensor) -- version of input batch perturbed by data
        gradient.
    """
    sign_batch_grad = batch_grad.sign()
    perturbed_batch = batch + epsilon * sign_batch_grad
    perturbed_batch = torch.clamp(perturbed_batch, min, max)
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
    
    batch.requires_grad = True

    output = model(batch)
    loss = loss_fcn(output, target)

    model.zero_grad()
    loss.backward()

    batch_grad = batch.grad.copy()
    batch.grad.zero_() # Not sure if setting requires grad to False zeroes out grads
    batch.requires_grad = False

    return batch_grad
    
def cw_attack(batch, target, model, device, lr, num_iter, c,
              min, max):
    """
    Generate adversarial versions of the given batch using an 
    untargeted L2 Carlini-Wagner attack.

    Keyword arguments:
    > batch (tensor) -- Collection of images to attack.
    > target (tensor) -- Ground truth target classes for each image in
        input batch.
    > model (nn.Module) -- Model with respect to which the attack is being done.
    > device (torch.device) -- Device for model computation
    > lr (float) -- Learning rate for PGD
    > num_iter (int) -- The number of iterations to perform PGD
    > c (float) -- The weight constant for the CW objective function
    > min (float) -- The minimum value that can occur in an image. 
    > max (float) -- The maximum value that can occur in an image.

    Return value: cw_batch
    > cw_batch (tensor) -- The adversarial batch generated.
    """
    
    # Initalize perturbation randomly
    perturbation = torch.randn_like(batch, requires_grad=True)

    # Begin PGD
    for iter in range(num_iter):
        # Compute loss
        loss = torch.sum(torch.pow(perturbation, 2))
        loss += c * cw_objective(model, batch, perturbation, target)

        # Perform backward pass
        model.zero_grad()
        perturbation.grad.zero_()
        loss.backward()

        # Perform PGD
        with torch.no_grad():
            perturbation -= lr * perturbation.grad
            perturbation = torch.clamp(batch + perturbation, min, max) - batch

    # Get perturbed batch
    with torch.no_grad():
        cw_batch = batch + perturbation

    return cw_batch

def cw_objective(model, batch, perturbation, target):
    """
    Computes the objective function used for untargeted L2 
    CW attacks. For further details, check page 9 of 
    Carlini and Wagner's paper "Towards Evaluating the Robustness
    of Neural Networks."

    Keyword arguments:
    > model (nn.Module) -- The model on which the attack is being performed.
    > batch (tensor) -- The batch to be perturbed.
    > perturbation (tensor) -- Adversarial perturbation of the same shape as batch.
    > target (tensor) -- The ground truth classes

    Return value: loss
    > loss (float) -- The loss from the L2 CW objective
    """
        
    scores = model(batch + perturbation)
    N, C = scores.shape

    true_scores = scores.gather(1, target.view(-1,1)).squeeze()
    scores[torch.arange(N), target] = torch.min(scores, 1)[0]
    max_scores = torch.max(scores, 1)[0]

    loss = torch.sum(torch.clamp(true_scores - max_scores, max=0))
    return loss
