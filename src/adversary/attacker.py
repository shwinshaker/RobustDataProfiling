#!./env python
# https://adversarial-ml-tutorial.org/adversarial_training/

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import accuracy, AverageMeter, GroupMeter, alignment, criterion_r
from ..preprocess import dataset_stats
from ..utils import DeNormalizer # used for autoattack training only
from .aa_attacker import AAAttacker # used for autoattack training only

from scipy.special import gammainc # For generation of uniform spherical perturbation

__all__ = ['attack', 'fgsm', 'ad_test', 'scale_step']

def scale_step(v, dataset, device='cpu'):
    # scale the epsilon based on stats in each channel
    n_channel = len(dataset_stats[dataset]['std'])
    std = torch.tensor(dataset_stats[dataset]['std']).view(n_channel, 1, 1).to(device)
    return v / 255. / std

def get_range(mean, std):
    n_channel = len(mean)
    mean_ = torch.tensor(mean).view(n_channel, 1, 1)
    std_ = torch.tensor(std).view(n_channel, 1, 1)
    return {'upper': (1 - mean_) / std_, 
            'lower': (0 - mean_) / std_}

dataset_range = dict([(dataset, get_range(dataset_stats[dataset]['mean'],
                                          dataset_stats[dataset]['std'])) for dataset in dataset_stats])

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def normalize(tensor):
    # scaled by l2 norm
    tol = 1e-8
    return tensor / (tensor.view(tensor.size(0), -1).norm(dim=1).view(tensor.size(0), 1, 1, 1) + tol)

def rand_sphere(eps, size, device=None, norm='linf', requires_grad=True):
    # using .data to avoid loading computation graph of delta
    if norm == 'linf':
        delta = torch.rand(size, requires_grad=requires_grad, device=device) # range = (0, 1] 
        delta.data = delta.data * 2 * eps - eps # make linf norm = eps

    elif norm == 'l2':
        delta = torch.randn(size, requires_grad=requires_grad, device=device)
        dim = torch.prod(torch.as_tensor(delta.size()[1:]))
        s2 = delta.view(delta.size(0), -1).norm(dim=1) ** 2
        base = (gammainc(dim.float() / 2.0, s2.cpu().numpy() / 2.0) ** (1 / dim.float())).to(device)
        base /= torch.sqrt(s2)
        delta.data = delta.data * base.repeat(dim, 1).T.view(delta.size()) * eps

    else:
        raise KeyError('Unexpected norm: %s' % norm)

    return delta

def compute_perturb(grad, norm='linf'):
    if norm == 'linf':
        return grad.sign()
    elif norm == 'l2':
        return normalize(grad)
    else:
        raise KeyError('Unexpected norm option: %s' % norm)
    
def fgsm(net, criterion, X, y, eps=0.1, norm='linf', rand_init=False, is_clamp=True, target=None, config=None):
    """ 
        Generate FGSM adversarial examples on the examples X
            # fgsm (single step)
            # pgd (multiple step)
            # CW (optimize difference between correct and incorrect logits)
    """
    # aligned with FAST
    net.train() 

    if is_clamp:
        delta_lower = dataset_range[config.dataset]['lower'].to(config.device) - X
        delta_upper = dataset_range[config.dataset]['upper'].to(config.device) - X

    if rand_init:
        delta = rand_sphere(eps, X.size(), device=config.device, norm=norm, requires_grad=True)
        if is_clamp:
            delta.data = clamp(delta, delta_lower, delta_upper)
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    # If label and targeted class are different, use targeted attack, sign = -1, gradient descent
    # If label and targeted class are same, use untargeted attack, sign = 1, gradient ascent
    signs = torch.tensor(1).repeat(X.size(0), 1, 1, 1).to(config.device)
    if target is not None:
        signs = ((y == target) * 2 - 1).view(X.size(0), 1, 1, 1).to(config.device)
        y = torch.tensor(target).repeat(X.size(0)).long().to(config.device)

    loss = criterion(net(X + delta), y)
    loss.backward()
    grad = delta.grad.detach()
    grad = compute_perturb(grad)

    delta.data = delta + eps * grad * signs
    delta.data = clamp(delta, -eps, eps)
    delta.data = clamp(delta, delta_lower, delta_upper)
    X_ad = X + delta.detach()

    net.eval()

    if is_clamp:
        return clamp(X_ad,
                     dataset_range[config.dataset]['lower'].to(config.device),
                     dataset_range[config.dataset]['upper'].to(config.device))
    return X_ad


def pgd_linf(net, criterion, X, y, eps=0.1, alpha=0.02, num_iter=5, rand_init=False, target=None, get_steps=False, config=None):
    """ 
        Construct PGD adversarial examples on the examples X
    """

    delta_lower = dataset_range[config.dataset]['lower'].to(config.device) - X
    delta_upper = dataset_range[config.dataset]['upper'].to(config.device) - X

    # use random start by default (aligned with Fast paper)
    if rand_init:
        delta = rand_sphere(eps, X.size(), device=config.device, norm='linf', requires_grad=True)
        delta.data = clamp(delta, delta_lower, delta_upper)
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    # If label and targeted class are different, use targeted attack, sign = -1, gradient descent
    # If label and targeted class are same, use untargeted attack, sign = 1, gradient ascent
    signs = torch.tensor(1).repeat(X.size(0), 1, 1, 1).to(config.device)
    if target is not None:
        # signs = ((y == target) * 2 - 1).view(X.size(0), 1, 1, 1).to(config.device)
        # y = torch.tensor(target).repeat(X.size(0)).long().to(config.device)

        signs = torch.tensor(-1).repeat(X.size(0), 1, 1, 1).to(config.device)
        y_ = torch.tensor(target).repeat(X.size(0)).long().to(config.device)
        y_[y == target] = 9 # truck
        y = y_
        
    if get_steps:
        step_counts = torch.zeros(X.size(0)).to(config.device)
    for t in range(num_iter):
        outputs_ad = net(X + delta)
        if get_steps:
            step_counts += (outputs_ad.max(1)[1] == y).float()

        loss = criterion(outputs_ad, y)
        loss.backward()
        delta.data = clamp(delta + alpha * delta.grad.detach().sign() * signs, -eps, eps)
        delta.data = clamp(delta, delta_lower, delta_upper)
        delta.grad.zero_()

    X_ad = clamp(X + delta.detach(),
                 dataset_range[config.dataset]['lower'].to(config.device),
                 dataset_range[config.dataset]['upper'].to(config.device))

    if get_steps:
        return X_ad, step_counts
    return X_ad

def attack(net, criterion, X, y, weight=None, adversary='fgsm', eps=0.1, pgd_alpha=0.02, pgd_iter=5, norm='linf', target=None, get_steps=False, randomize=False, is_clamp=True, config=None):

    if weight is not None:
        eps = eps.repeat(X.size(0), 1, 1, 1) * weight.view(X.size(0), 1, 1, 1).to(config.device)

    if adversary == 'fgsm':
        if get_steps:
            raise RuntimeError('step count is not applicable to single-step attack!')
        return fgsm(net, criterion, X, y, eps=eps, norm=norm, rand_init=randomize, is_clamp=is_clamp, target=target, config=config)
    elif adversary == 'pgd':
        return pgd_linf(net, criterion, X, y, eps=eps, alpha=pgd_alpha, num_iter=pgd_iter, rand_init=randomize, target=target, get_steps=get_steps, config=config)
    elif adversary.lower() == 'aa':
        # While using autoattack, 'eps' is neglected as it is scaled, instead used the value in config
        denormalize = DeNormalizer(dataset_stats[config.dataset]['mean'],
                                   dataset_stats[config.dataset]['std'],
                                   X.size(1), config.device)
        attacker = AAAttacker(net=net,
                              eps=config.eps_, # use unscaled eps
                              normalize=True,
                              mode='fast',
                              path='.',
                              device=config.device,
                              data_dir=None)
        X_ = denormalize(X)
        X_ad, _ = attacker.evaluate(x_test=X_, y_test=y)
        X_ad = attacker._normalize(X_ad)
        return X_ad
    else:
        raise KeyError(adversary)


def ad_test(dataloader, net, criterion, config, classes=None):
    losses = AverageMeter()
    top1 = AverageMeter()

    for i, tup in enumerate(dataloader, 0):
        if len(tup) == 2:
            inputs, labels = tup
        else:
            inputs, labels, _ = tup
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        inputs_ad = attack(net, criterion, inputs, labels,
                           adversary=config.ad_test, eps=config.eps_test, pgd_alpha=config.pgd_alpha_test, pgd_iter=config.pgd_iter_test,
                           config=config)  
        outputs_ad = net(inputs_ad)
        loss_ad = criterion(outputs_ad, labels)
        prec_ad, = accuracy(outputs_ad.data, labels.data)

        losses.update(loss_ad.item(), inputs.size(0))        
        top1.update(prec_ad.item(), inputs.size(0))

    return losses.avg, top1.avg

