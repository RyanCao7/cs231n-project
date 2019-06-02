import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def attack_batch(batch, target, model, loss_fcn, attack_name='FGSM', 
                 device=torch.device('cpu'), epsilon=0.3, alpha=0.05,
                 lr=10.0, num_iter=100, c=100.0, min_pix=0, max_pix=1):
    '''
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
        taken from Defense GAN paper.
    > min_pix (float) -- Minimum value that can occur in an image. Default value
        assumes no normalization has taken place.
    > max_pix (float) -- Maximum value that can occur in an image. Default value
        assumes no normalization has taken place.

    Return value: adversarial_batch
    > adversarial_batch (tensor) -- adversarial version of input batch.
    '''

    batch = batch.to(device)
    target = target.to(device)
    model.eval()

    # We are doing untargeted, so we are going to ignore the 'target' classes
    if attack_name == 'FGSM':
        return fgsm_attack(batch, 
                           epsilon, 
                           get_batch_grad(batch, model, loss_fcn, device, target=None),
                           min_pix, 
                           max_pix)
    elif attack_name == 'RAND_FGSM':
        noisy_batch = batch + alpha * torch.sign(torch.randn_like(batch))
        return fgsm_attack(noisy_batch, 
                           epsilon - alpha, 
                           get_batch_grad(noisy_batch, model, loss_fcn, device, target=None),
                           min_pix, 
                           max_pix)
    elif attack_name == 'CW':
#         return cw_attack(batch, target, model, device, lr, num_iter, c, min_pix, max_pix)
        return reformulated_cw_attack_adam(batch, target, model, device, lr, num_iter, c)
    else:
        raise Exception('Error: attack_name must be one of {\'FGSM\', \'RAND_FGSM\', '
                        '\'CW\'}.')

    
def fgsm_attack(batch, epsilon, batch_grad, min_pix, max_pix):
    '''
    Performs the Fast Gradient Sign Method (FGSM) attack as described by
    Goodfellow et al. in "Explaining and Harnessing Adversarial Examples".

    Keyword arguments:
    > batch (tensor) -- Collection of images to attack.
    > epsilon (float) -- Weight for data gradient used.
    > batch_grad (tensor) -- Loss gradient with respect to batch.
    > max_pix (float) -- The maximum value possible in a valid image. 
    > min_pix (float) -- The minimum value possible in a valid image. 

    Return value: perturbed_batch
    > perturbed_batch (tensor) -- version of input batch perturbed by data
        gradient.
    '''
    sign_batch_grad = batch_grad.sign()
    perturbed_batch = batch + epsilon * sign_batch_grad
    perturbed_batch = torch.clamp(perturbed_batch, min_pix, max_pix)
    return perturbed_batch.detach()


def get_batch_grad(batch, model, loss_fcn, device, target=None):
    '''
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
    '''

    batch = batch.clone().detach().to(torch.float).requires_grad_(True) 

    if target is None:
        # Using model predictions as ground truth to avoid label leaking
        # See https://arxiv.org/abs/1611.01236 for more details
        _, target = torch.max(model(batch), 1)

    output = model(batch)
    loss = loss_fcn(output, target)

    model.zero_grad()
    loss.backward()

    batch_grad = batch.grad.clone()
    return batch_grad
    
    
def cw_attack(batch, target, model, device, lr, num_iter, c,
              min_pix, max_pix):
    '''
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
    > min_pix (float) -- The minimum value that can occur in an image. 
    > max_pix (float) -- The maximum value that can occur in an image.

    Return value: cw_batch
    > cw_batch (tensor) -- The adversarial batch generated.
    '''
    # Initalize perturbation randomly
    perturbation = torch.randn_like(batch, requires_grad=True)

    # Begin PGD
    for iter in range(num_iter):
        # So gradients don't stack up + to not modify a tensor we backpropagate through
        temp_perturbation = perturbation.detach().clone().requires_grad_()
        
        # Compute loss
        t_loss = torch.sum(torch.pow(temp_perturbation, 2))
        loss = t_loss + c * cw_objective(model, batch, temp_perturbation, target)

        # Perform backward pass
        model.zero_grad() # I don't think we need this, since the model is in eval() mode... right???
        loss.backward()

        # Perform PGD
        with torch.no_grad():
            perturbation -= lr * temp_perturbation.grad
            perturbation = torch.clamp(batch + perturbation, min_pix, max_pix) - batch

    # Get perturbed batch
    with torch.no_grad():
        cw_batch = batch + perturbation

    return cw_batch.detach()


def w_to_delta(w, batch):
    '''
    Change-of-basis from w to delta.
    '''
    return 0.5 * (torch.tanh(w) + 1) - batch


def reformulated_cw_attack(batch, target, model, device, lr, num_iter, c):
    '''
    Reformulated version of cw attack without clipping.
    Optimizes w rather than delta directly - see page 7
    of CW attack paper for more details!
    '''
    
    # TODO: Perhaps this will be slightly more efficient...?
    batch.requires_grad = False
    
    # ~N(0, 1) - Gaussian with \mu = 0; \sigma = 1.
    # No grad needed - will do manual updates (TODO: is this slow?)
    w = torch.randn_like(batch)
    
    for i in range(num_iter):
        # Create detached copy of w to backprop through
        temp_w = w.detach().clone().requires_grad_()
        
        # Get raw logits from model
        delta = w_to_delta(temp_w, batch)
        logits = model(batch + delta)

        # Compute cw minimization objective (L2)
        perturbation_term = torch.sqrt(torch.sum(delta ** 2))
        objective = perturbation_term + c * cw_objective(model, batch, delta, target)
        
        # Backpropagate
        objective.backward()
        
        # Manually perform gradient descent
        with torch.no_grad():
            w -= lr * temp_w.grad
            
    return (batch + w_to_delta(w, batch)).detach()
    
    
def reformulated_cw_attack_adam(batch, target, model, device, lr, num_iter, c):
    '''
    Reformulated version of cw attack using
    Adam optimizer rather than SGD.
    '''
    # TODO: Perhaps this will be slightly more efficient...?
    batch.requires_grad = False
    
    # ~N(0, 1) - Gaussian with \mu = 0; \sigma = 1.
    w = torch.randn_like(batch, requires_grad=True)
    adam = optim.Adam([w], lr=lr)
    
    for i in range(num_iter):
        
        # Get raw logits from model
        delta = w_to_delta(w, batch)
        logits = model(batch + delta)

        # Compute cw minimization objective (L2)
        perturbation_term = torch.sqrt(torch.sum(delta ** 2))
        objective = perturbation_term + c * cw_objective(model, batch, delta, target)
        
        # Backpropagate
        if w.grad is not None:
            w.grad.data.zero_()
        objective.backward()
        
        # Backpropagate
        adam.step()
            
    return (batch + w_to_delta(w, batch)).detach()
    

def cw_objective(model, batch, perturbation, target):
    '''
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
    '''
    
    # Computes raw logits
    scores = model(batch + perturbation)
    N, C = scores.shape
    
    # Reshape into 1-D
    true_scores = scores.gather(1, target.view(-1, 1)).squeeze()
    
    # Prevent re-assignment of scores
    scores_copy = scores.detach().clone()
    scores_copy[torch.arange(N), target] = torch.min(scores, 1)[0]
    max_scores = torch.max(scores_copy, 1)[0]

    loss = torch.sum(torch.clamp(true_scores - max_scores, max=0))
    return loss
