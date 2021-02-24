#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attacker import dataset_range, clamp, rand_sphere, attack

__all__ = ['trades_loss']

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
