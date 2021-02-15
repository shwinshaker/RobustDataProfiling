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

__all__ = ['attack', 'gaussian', 'fgsm', 'ad_test', 'scale_step']

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
        """
            # delta = torch.randn_like(X)
            # delta = normalize(delta) * eps

            incorrect, l2-scaled gaussian distribution generates uniform distribution on the sphere, not inside the sphere
            for uniform distribution inside the sphere, refer to `random_sphere` in `adversarial-robustness-toolbox/art/utils.py`
                and https://www.mathworks.com/matlabcentral/fileexchange/9443-random-points-in-an-n-dimensional-hypersphere
        """

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
    

def gaussian(X, eps=0.1, norm='linf', config=None):
    delta_lower = dataset_range[config.dataset]['lower'].to(config.device) - X
    delta_upper = dataset_range[config.dataset]['upper'].to(config.device) - X

    delta = rand_sphere(eps, X.size(), device=config.device, norm=norm, requires_grad=False)
    delta = clamp(delta, delta_lower, delta_upper)
    return X + delta

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


def fgsm_manifold(net, X, y, eps=0.1, norm='linf', rand_init=False, config=None):
    """
        try manifold attack and see any difference
            Notice the difference wrt to `on manifold adversarial attack`
    """
    delta_lower = dataset_range[config.dataset]['lower'].to(config.device) - X
    delta_upper = dataset_range[config.dataset]['upper'].to(config.device) - X

    if rand_init:
        delta = rand_sphere(eps, X.size(), device=config.device, norm=norm, requires_grad=True)
        delta.data = clamp(delta, delta_lower, delta_upper)
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    log_softmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    def kl_loss(outputs_ad, outputs):
        return criterion_kl(log_softmax(outputs_ad), softmax(outputs))

    loss = kl_loss(net(X + delta), net(X))
    loss.backward()
    grad = delta.grad.detach()
    grad = compute_perturb(grad)
    """
        FGSM is slightly different from one-step PGD when using random initialization
        Delta is the gradient at the random initialized point
        For PGD,
        Delta is the initial perturbation + the gradient at the random initialized point
    """

    return clamp(X + eps * grad,
                 dataset_range[config.dataset]['lower'].to(config.device),
                 dataset_range[config.dataset]['upper'].to(config.device))


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


def pgd_linf_manifold(net, X, y, eps=0.1, alpha=0.02, num_iter=5, rand_init=False, config=None):

    delta_lower = dataset_range[config.dataset]['lower'].to(config.device) - X
    delta_upper = dataset_range[config.dataset]['upper'].to(config.device) - X

    if rand_init:
        delta = rand_sphere(eps, X.size(), device=config.device, norm='linf', requires_grad=True)
        delta.data = clamp(delta, delta_lower, delta_upper)
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    log_softmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    def kl_loss(outputs_ad, outputs):
        return criterion_kl(log_softmax(outputs_ad), softmax(outputs))

    for t in range(num_iter):
        loss = kl_loss(net(X + delta), net(X))
        loss.backward()
        delta.data = clamp(delta + alpha * delta.grad.detach().sign(), -eps, eps)
        delta.data = clamp(delta, delta_lower, delta_upper)
        delta.grad.zero_()

    return clamp(X + delta.detach(),
                 dataset_range[config.dataset]['lower'].to(config.device),
                 dataset_range[config.dataset]['upper'].to(config.device))



from collections.abc import Iterable
def pgd_linf_custom(net, criterion, X, y, eps=0.1, alpha=0.02, num_iter=5,
                    rand_init=False, config=None):
    '''
    Custom PGD:
        Allow examples-wise customized PGD attack
        Available customized parameters:
            * eps: Epsilon
            * num_iter: Number of iterations
            * alpha: Step size
        Other parameters:
            * rand_init: initialize the perturbation with random noise?
            * config: Configure settings
    '''

    # sanity check
    # if not isinstance(eps, Iterable):
    if eps.size() == (3, 1, 1):
        eps = eps.repeat(X.size(0), 1, 1, 1)
    else:
        assert(eps.size(0) == X.size(0)), eps.size()

    if alpha.size() == (3, 1, 1):
        alpha = alpha.repeat(X.size(0), 1, 1, 1)
    else:
        assert(alpha.size(0) == X.size(0)), alpha.size()

    if not isinstance(num_iter, Iterable):
        num_iter = torch.tensor(num_iter).repeat(X.size(0))
    else:
        assert(num_iter.size(0) == X.size(0)), num_iter.size()

    delta_lower = dataset_range[config.dataset]['lower'].to(config.device) - X
    delta_upper = dataset_range[config.dataset]['upper'].to(config.device) - X

    if rand_init:
        delta = rand_sphere(eps, X.size(), device=config.device, norm='linf',
                            requires_grad=False)
        delta.data = clamp(delta, delta_lower, delta_upper)
    else:
        delta = torch.zeros_like(X, requires_grad=False)

    iter_dec = num_iter.clone()
    iter_inc = torch.zeros(len(y))

    while torch.any(iter_dec > 0):
        iter_index = (iter_dec > 0).nonzero().squeeze()
        # print(torch.max(iter_dec).item(), len(iter_index))
        iter_delta = torch.zeros_like(delta[iter_index], requires_grad=True)
        iter_delta.data = delta[iter_index]
        iter_inc[iter_index] += 1 # For sanity check

        # iter_delta.requires_grad_()
        loss = criterion(net(X[iter_index] + iter_delta), y[iter_index])
        loss.backward()
    
        iter_delta.data = clamp(iter_delta + alpha[iter_index] * iter_delta.grad.detach().sign(),
                                -eps[iter_index], eps[iter_index])
        iter_delta.data = clamp(iter_delta, delta_lower[iter_index], delta_upper[iter_index])
        delta[iter_index] = iter_delta

        iter_dec[iter_index] = iter_dec[iter_index] - 1

    # sanity check
    assert(torch.all(iter_inc == num_iter))

    X_ad = clamp(X + delta.detach(),
                 dataset_range[config.dataset]['lower'].to(config.device),
                 dataset_range[config.dataset]['upper'].to(config.device))

    return X_ad


def attack(net, criterion, X, y, weight=None, adversary='fgsm', eps=0.1, pgd_alpha=0.02, pgd_iter=5, norm='linf', target=None, get_steps=False, randomize=False, is_clamp=True, config=None):

    if weight is not None:
        eps = eps.repeat(X.size(0), 1, 1, 1) * weight.view(X.size(0), 1, 1, 1).to(config.device)

    if adversary == 'gaussian':
        return gaussian(X, eps=eps, norm=norm, config=config)
    elif adversary == 'fgsm':
        if get_steps:
            raise RuntimeError('step count is not applicable to single-step attack!')
        return fgsm(net, criterion, X, y, eps=eps, norm=norm, rand_init=randomize, is_clamp=is_clamp, target=target, config=config)
    elif adversary == 'pgd':
        return pgd_linf(net, criterion, X, y, eps=eps, alpha=pgd_alpha, num_iter=pgd_iter, rand_init=randomize, target=target, get_steps=get_steps, config=config)
    elif adversary == 'pgd_custom':
        return pgd_linf_custom(net, criterion, X, y, eps=eps, alpha=pgd_alpha, num_iter=pgd_iter, rand_init=randomize, config=config)
    elif adversary == 'fgsm_manifold':
        return fgsm_manifold(net, X, y, eps=eps, norm=norm, rand_init=randomize, target=target, config=config)
    elif adversary == 'pgd_manifold':
        return pgd_linf_manifold(net, X, y, eps=eps, alpha=pgd_alpha, num_iter=pgd_iter, rand_init=randomize, target=target, get_steps=get_steps, config=config)
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

    robust_scores = dict([(m, AverageMeter()) for m in config.robust_metrics])
    if hasattr(config, 'class_eval') and config.class_eval:
        top1_class = GroupMeter(classes)

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

        if hasattr(config, 'class_eval') and config.class_eval:
            top1_class.update(outputs_ad, labels)

        if 'rb' in config.robust_metrics:
            """
                Ratio of examples that change labels after attack, regardless of the true labels
            """
            outputs = net(inputs)
            # default pytorch cross-entropy only allows long type targets
            # loss_r = criterion_r(outputs_ad, outputs, c=criterion)  # not debugged, deprecated
            prec_r = alignment(outputs_ad.data, outputs.data)
    
            # robust_scores['rb']['loss'].update(loss_r.item(), inputs.size(0))        
            robust_scores['rb'].update(prec_r.item(), inputs.size(0))

    # robust_score_avg = 
    extra_metrics = dict()
    if config.robust_metrics:
        extra_metrics['rb_metric'] = dict([(m, robust_scores[m].avg) for m in robust_scores])
    if hasattr(config, 'class_eval') and config.class_eval:
        extra_metrics['class_acc'] = top1_class.output_group()
    
    return losses.avg, top1.avg, extra_metrics # robust_score_avg, top1_class

