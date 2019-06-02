import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        # return fgsm_attack(batch, 
        #                    epsilon, 
        #                    get_batch_grad(batch, model, loss_fcn, device, target=None),
        #                    min_pix, 
        #                    max_pix)
        return fast_gradient_method(model, 
                                    batch, 
                                    epsilon, 
                                    np.inf, 
                                    clip_min=min_pix, 
                                    clip_max=max_pix,
                                    sanity_checks=True)
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


def fast_gradient_method(model_fn, x, eps, ord,
                         clip_min=None, clip_max=None, y=None, targeted=False, sanity_checks=False):
  """
  PyTorch implementation of the Fast Gradient Method.
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
  :param ord: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components.
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
  :return: a tensor for the adversarial example
  """
  if ord not in [np.inf, 1, 2]:
    raise ValueError("Norm order must be either np.inf, 1, or 2.")

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
    asserts.append(assert_ge)

  if clip_max is not None:
    assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
    asserts.append(assert_le)

  # x needs to be a leaf variable, of floating point type and have requires_grad being True for
  # its grad to be computed and stored properly in a backward call
  x = x.clone().detach().to(torch.float).requires_grad_(True)
  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    _, y = torch.max(model_fn(x), 1)

  # Compute loss
  loss_fn = torch.nn.CrossEntropyLoss()
  loss = loss_fn(model_fn(x), y)
  # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
  if targeted:
    loss = -loss

  # Define gradient of loss wrt input
  loss.backward()
  optimal_perturbation = optimize_linear(x.grad, eps, ord)

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + optimal_perturbation

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    # We don't currently support one-sided clipping
    assert clip_min is not None and clip_max is not None
    adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x


def optimize_linear(grad, eps, ord=np.inf):
  """
  Solves for the optimal input to a linear function under a norm constraint.

  Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)

  :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
  :param eps: float. Scalar specifying size of constraint region
  :param ord: np.inf, 1, or 2. Order of norm constraint.
  :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
  """

  red_ind = list(range(1, len(grad.size())))
  avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
  if ord == np.inf:
    # Take sign of gradient
    optimal_perturbation = torch.sign(grad)
  elif ord == 1:
    abs_grad = torch.abs(grad)
    sign = torch.sign(grad)
    red_ind = list(range(1, len(grad.size())))
    abs_grad = torch.abs(grad)
    ori_shape = [1]*len(grad.size())
    ori_shape[0] = grad.size(0)

    max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
    max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
    num_ties = max_mask
    for red_scalar in red_ind:
      num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
    optimal_perturbation = sign * max_mask / num_ties
    # TODO integrate below to a test file
    # check that the optimal perturbations have been correctly computed
    opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
    assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
  elif ord == 2:
    square = torch.max(
        avoid_zero_div,
        torch.sum(grad ** 2, red_ind, keepdim=True)
        )
    optimal_perturbation = grad / torch.sqrt(square)
    # TODO integrate below to a test file
    # check that the optimal perturbations have been correctly computed
    opt_pert_norm = optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
    one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + \
            (square > avoid_zero_div).to(torch.float)
    assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
  else:
    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                              "currently implemented.")

  # Scale perturbation to be the solution for the norm=eps rather than
  # norm=1 problem
  scaled_perturbation = eps * optimal_perturbation
  return scaled_perturbation

    
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
