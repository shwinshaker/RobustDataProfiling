#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attacker import dataset_range, clamp, rand_sphere, attack

__all__ = ['bce_loss', 'trades_loss', 'llr_loss', 'mart_loss', 'fat_loss', 'gairat_loss']

def kl_loss(outputs_ad, outputs, reduction='mean'):
    if reduction == 'mean':
        # sumed and averaged over batch dimension
        return F.kl_div(F.log_softmax(outputs_ad, dim=1),
                        F.softmax(outputs, dim=1),
                        reduction='batchmean')
    if reduction == 'none':
        # still need to manually sum over the class dimensions
        loss =  F.kl_div(F.log_softmax(outputs_ad, dim=1),
                         F.softmax(outputs, dim=1),
                         reduction='none')
        return loss.sum(dim=1)
    raise KeyError(reduction)


def trades_loss(net, X, y, weights=None,
                eps=0.1, alpha=0.02, num_iter=5, norm='linf',
                rand_init=True, config=None, beta=6.0):

    net.eval()

    if norm != 'linf':
        raise NotImplementedError

    delta_lower = dataset_range[config.dataset]['lower'].to(config.device) - X
    delta_upper = dataset_range[config.dataset]['upper'].to(config.device) - X

    if rand_init:
        delta = rand_sphere(eps, X.size(), device=config.device, norm='linf', requires_grad=True)
        delta.data = clamp(delta, delta_lower, delta_upper)
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = kl_loss(net(X + delta), net(X))
        # loss = F.cross_entropy(net(X + delta), y)
        loss.backward()
        delta.data = clamp(delta + alpha * delta.grad.detach().sign(), -eps, eps)
        delta.data = clamp(delta, delta_lower, delta_upper)
        delta.grad.zero_()

    # X_ad = clamp(X + delta.detach(),
    #              dataset_range[config.dataset]['lower'].to(config.device),
    #              dataset_range[config.dataset]['upper'].to(config.device))

    net.train()

    outputs = net(X)
    outputs_ad = net(X + delta)
    loss = F.cross_entropy(outputs, y, reduction='none')
    loss_ad = kl_loss(outputs_ad, outputs, reduction='none')
    # loss_ad = F.cross_entropy(net(X + delta), y, reduction='none')
    assert(loss.size() == loss_ad.size()), (loss.size(), loss_ad.size())
    if 'alpha' in weights:
        # replace F.softmax(outputs) as the pre-calculated probabilities (but needs to be 10-dimensional?)
        raise NotImplementedError('per-sample weighting in trades not supported yet. #TODO')
    loss += loss_ad * beta

    loss = loss.mean()
    return loss, outputs_ad


def bce_loss(outputs, y, weights=None, reduction='none'):
    """
        bce loss for multiple dimensions
        essentially a margin-maximization loss
        * see mart paper

        l = - y \log p - (1-y) \log (1-p)
    """
    probs = F.softmax(outputs, dim=1)
    tmp1 = torch.argsort(probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    # if prediction is correct, probability of the true class is the highest, use the second highest
    # if prediction is wrong, use the highest probability
    return F.cross_entropy(outputs, y, reduction=reduction) + F.nll_loss(torch.log(1.0001 - probs + 1e-12), new_y, reduction=reduction)


def mart_loss(net, X, y, weights=None,
              eps=0.1, alpha=0.02, num_iter=5, norm='linf',
              rand_init=True, config=None, beta=6.0):

    net.eval()

    if norm != 'linf':
        raise NotImplementedError

    delta_lower = dataset_range[config.dataset]['lower'].to(config.device) - X
    delta_upper = dataset_range[config.dataset]['upper'].to(config.device) - X

    if rand_init:
        delta = rand_sphere(eps, X.size(), device=config.device, norm='linf', requires_grad=True)
        delta.data = clamp(delta, delta_lower, delta_upper)
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        # loss = kl_loss(net(X + delta), net(X))
        loss = F.cross_entropy(net(X + delta), y)
        loss.backward()
        delta.data = clamp(delta + alpha * delta.grad.detach().sign(), -eps, eps)
        delta.data = clamp(delta, delta_lower, delta_upper)
        delta.grad.zero_()

    net.train()

    outputs = net(X)
    outputs_ad = net(X + delta)
    # loss = F.cross_entropy(outputs_ad, y, reduction='none') # if CE
    loss = bce_loss(outputs_ad, y, reduction='none') # if BCE

    loss_ad = kl_loss(outputs_ad, outputs, reduction='none')

    # per sample weighting
    if 'alpha' in weights:
        prob = weights['alpha'].to(config.device)
    else:
        prob = torch.gather(F.softmax(outputs, dim=1), 1, y.unsqueeze(1).long()).squeeze()
    assert(loss.size() == loss_ad.size() == prob.size()), (loss.size(), loss_ad.size(), prob.size())

    # official
    loss += loss_ad * beta * torch.clamp(1. - prob, 0., 1.)

    # scaled
    # loss *= (0.1 + prob)
    # loss += loss_ad * (1.0 - prob)

    loss = loss.mean()
    return loss, outputs_ad

from .earlystop import earlystop
def fat_loss(net, X, y, weights=None,
             eps=0.1, alpha=0.02, num_iter=5, norm='linf',
             rand_init=True, config=None, tau=0):

    # needs to reorder weights along with early stopping
    # Get friendly adversarial training data via early-stopped PGD
    inputs_ad, ad_steps = earlystop(net,
                                    X,
                                    y,
                                    epsilon=eps,
                                    step_size=alpha,
                                    perturb_steps=num_iter,
                                    tau=tau,
                                    rand_init=rand_init,
                                    randominit_type=norm,
                                    loss_fn='cent',
                                    omega=0.001,
                                    config=config) # args.omega)

    net.train()
    outputs_ad = net(inputs_ad)
    loss = F.cross_entropy(outputs_ad, y, reduction='none')

    loss = loss.mean()
    return loss, outputs_ad, ad_steps

def gairat_weight_function(kappa, num_iter, lamb=0):
    standard_kappa = 1. - kappa / num_iter * 2. # [-1, 1]
    weight = torch.tanh(standard_kappa * 5. + lamb) # [-1, 1]
    weight = (weight + 1.) / 2. # [0, 1]
    weight /= torch.sum(weight) # scaled
    return weight

def gairat_loss(net, X, y, weights=None,
                eps=0.1, alpha=0.02, num_iter=5, norm='linf',
                rand_init=True, config=None, tau=0):

    net.eval()
    ctr = nn.CrossEntropyLoss()
    inputs_ad, ad_steps = attack(net, ctr, X, y, weight=None,
                                 adversary='pgd',
                                 eps=config.eps,
                                 pgd_alpha=config.pgd_alpha,
                                 pgd_iter=config.pgd_iter,
                                 randomize=config.rand_init,
                                 target=None,
                                 get_steps=True,
                                 config=config)

    net.train()
    outputs_ad = net(inputs_ad)
    loss = F.cross_entropy(outputs_ad, y, reduction='none')

    weight = gairat_weight_function(ad_steps, num_iter).to(config.device)
    assert(weight.size() == loss.size()), (weight.size(), loss.size())
    loss = (loss * weight).sum()

    return loss, outputs_ad, ad_steps


def llr_loss(net, X, y, eps=0.1, alpha=0.02, num_iter=5, norm='linf', rand_init=True, config=None, lambd=4.0, mu=3.0):

    net.eval()

    if norm != 'linf':
        raise NotImplementedError

    delta_lower = dataset_range[config.dataset]['lower'].to(config.device) - X
    delta_upper = dataset_range[config.dataset]['upper'].to(config.device) - X

    if rand_init:
        delta = rand_sphere(eps, X.size(), device=config.device, norm='linf', requires_grad=True)
        delta.data = clamp(delta, delta_lower, delta_upper)
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    def model_grad(x):
        x.requires_grad_(True)
        lx = F.cross_entropy(net(x), y)
        lx.backward()
        ret = x.grad
        x.grad.zero_()
        x.requires_grad_(False)
        return ret

    def grad_dot(delta, grad, reduction='mean'):
        ret = torch.matmul(grad.flatten(start_dim=1),
                           delta.flatten(start_dim=1).T)
        if reduction == 'mean':
            return ret.mean()
        return ret

    def g(x, delta: torch.Tensor, grad, reduction='mean'):
        # ret = criterion(net(x+delta), y) - grad_dot(x, delta, model_grad)
        ret = F.cross_entropy(net(x + delta), y, reduction=reduction) \
            - F.cross_entropy(net(x), y, reduction=reduction) \
            - grad_dot(delta, grad, reduction=reduction)
        return ret.abs()

    X_grad = model_grad(X)

    for t in range(num_iter):
        loss = g(X, delta, X_grad)
        loss.backward()
        delta.data = clamp(delta + alpha * delta.grad.detach().sign(), -eps, eps)
        delta.data = clamp(delta, delta_lower, delta_upper)
        delta.grad.zero_()

    net.train()

    # # zero gradient
    # optimizer.zero_grad()
    # # calculate robust loss
    # outputs = net(x)
    # loss_natural = criterion(outputs, y)
    # if version == "sum":
    #     loss = loss_natural + lambd * g(x, delta, mg) + mu * grad_dot(x, delta, mg) * len(x)
    # else:
    #     loss = loss_natural + lambd * g(x, delta, mg) + mu * grad_dot(x, delta, mg)

    loss = (lambd * g(X, delta, X_grad, reduction=config.reduction) + mu * grad_dot(delta, X_grad, reduction=config.reduction)) / (lambd + mu)

    return loss
