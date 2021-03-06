#!./env python

import torch
import torch.nn as nn
from ..utils import Logger, AverageMeter, accuracy
from ..utils import ExampleTracker, AvgTracker
from . import attack, scale_step
from .loss import trades_loss
import time

__all__ = ['AdTrainer']

class AdTrainer:
    """
        wrapping class for adversary training, including logging
    """
    def __init__(self, loaders, net, optimizer, criterion=None, config=None, time_start=None):
        self.loaders = loaders
        self.net = net
        self.optimizer = optimizer # only used for getting current learning rate
        self.criterion = criterion
        self.config = config
        self.device = self.config.device
        self.time_start = time_start

        # target
        self.target = None
        if config.target is not None:
            self.target = loaders.class_to_idx[config.target]

        # scale epsilon (each channel is different because different range)
        ## save unscaled eps for auto attack
        config.eps_ = config.eps
        config.pgd_alpha_ = config.pgd_alpha
        config.eps = scale_step(config.eps, config.dataset, device=config.device)
        config.pgd_alpha = scale_step(config.pgd_alpha, config.dataset, device=config.device)
        print('scaled eps [train]:', config.eps, config.pgd_alpha)

        # sanity check and setup loss function
        self.__ad_setup()
        self.epoch = 0

    def _loss(self, inputs, labels, weights, epoch=None):
        # template
        pass

    def update(self, epoch, i):
        # make some logs

        if self.extra_metrics:
            self.extraLog.step(epoch, i)

        if self.config.exTrack:
            self.exLog.step(epoch)

        self.epoch = epoch + 1

    def reset(self, epoch):
        assert(epoch == self.epoch - 1), 'reset is not called after update!'

        ## reset some logger
        if self.extra_metrics:
            self.extraLog.reset()

        if self.config.exTrack:
            self.exLog.reset()

    def close(self):
        if self.extra_metrics:
            self.extraLog.close()


    def _clean_loss(self, inputs, labels, weights, epoch=None):
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
    
        # keep for record
        if self.extra_metrics:
            prec1, = accuracy(outputs.data, labels.data)
            self.extraLog.update({'Train-Loss': loss.mean().item(),
                                  'Train-Acc': prec1.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            self.exLog.update(outputs, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss.mean()

    def _ad_loss(self, inputs, labels, weights, epoch=None):

        # -------- clean loss
        loss = 0.
        # if pure ad loss and sample-wise alpha not enabled, don't have to do this part
        if self.config.alpha < 1.0:
            loss = self._clean_loss(inputs, labels, weights)

        # ------- ad loss
        self.net.eval()
        ctr = nn.CrossEntropyLoss() # Don't change the criterion in adversary generation part -- maybe change it later
        inputs_ad = attack(self.net, ctr, inputs, labels, weight=None,
                           adversary=self.config.adversary,
                           eps=self.config.eps,
                           pgd_alpha=self.config.pgd_alpha,
                           pgd_iter=self.config.pgd_iter,
                           randomize=self.config.rand_init,
                           target=self.target,
                           config=self.config)
        self.net.train()
        outputs_ad = self.net(inputs_ad)
        loss_ad = self.criterion(outputs_ad, labels)

        # -------- combine two loss
        loss *= (1 - self.config.alpha)
        loss += self.config.alpha * loss_ad
        loss = loss.mean()

        # -------- recording
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss

    def _trades_loss(self, inputs, labels, weights, epoch=None):
        loss, outputs_ad = trades_loss(self.net, inputs, labels, weights,
                                      eps=self.config.eps,
                                      alpha=self.config.pgd_alpha,
                                      num_iter=self.config.pgd_iter,
                                      norm='linf',
                                      rand_init=self.config.rand_init,
                                      config=self.config)

        # -------- recording
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            # Generate ad examples using PGD, otherwise not fair!
            outputs_ad = self.__get_pgd_ad(inputs, labels)
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss

    def __ad_setup(self):
        base_names = ['Epoch', 'Mini-batch', 'lr', 'Time-elapse(Min)']
        self.logger_e = Logger('log_extra.txt', title='log for deprecated metrics', resume=self.config.resume)

        if not self.config.adversary:
            self._loss = self._clean_loss

            # log for 'false' training loss and acc, aligned with previous work
            self.extra_metrics = ['Train-Loss', 'Train-Acc']
            self.extraLog = AvgTracker('log_extra',
                                       self.optimizer,
                                       metrics=self.extra_metrics,
                                       time_start=self.time_start,
                                       config=self.config)

            # log for sample robust correctness
            if self.config.exTrack:
                self.exLog = ExampleTracker(self.loaders, resume=self.config.resume)

            return


        if self.config.adversary in ['gaussian', 'fgsm', 'pgd', 'aa']:
            self._loss = self._ad_loss

            # log for 'false' training loss and acc, aligned with previous work
            self.extra_metrics = ['Train-Loss', 'Train-Acc', 'Train-Loss-Ad', 'Train-Acc-Ad']
            self.extraLog = AvgTracker('log_extra',
                                       self.optimizer,
                                       metrics=self.extra_metrics,
                                       time_start=self.time_start,
                                       config=self.config)

            # log for sample robust correctness
            if self.config.exTrack:
                self.exLog = ExampleTracker(self.loaders, resume=self.config.resume)

            return

        # other 
        if self.config.target:
            raise NotImplementedError('Targeted attack not supported! TODO..')

        if self.config.adversary == 'trades':
            self._loss = self._trades_loss

            # log for 'false' training loss and acc, aligned with previous work
            self.extra_metrics = ['Train-Loss-Ad', 'Train-Acc-Ad']
            self.extraLog = AvgTracker('log_extra',
                                       self.optimizer,
                                       metrics=self.extra_metrics,
                                       time_start=self.time_start,
                                       config=self.config)

            if self.config.exTrack:
                self.exLog = ExampleTracker(self.loaders, resume=self.config.resume)

            return

        raise KeyError('Unexpected adversary %s' % self.config.adversary)

    def __get_pgd_ad(self, inputs, labels):
        self.net.eval()
        ctr = nn.CrossEntropyLoss() # Don't change the criterion in adversary generation part -- maybe change it later
        inputs_ad = attack(self.net, ctr, inputs, labels, weight=None,
                           adversary='pgd',
                           eps=self.config.eps,
                           pgd_alpha=self.config.pgd_alpha,
                           pgd_iter=self.config.pgd_iter,
                           randomize=self.config.rand_init,
                           target=self.target,
                           config=self.config)
        # don't affect the training stats, do this in eval mode
        outputs_ad = self.net(inputs_ad)
        self.net.train()
        return outputs_ad

    def __get_lr(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        assert(len(lrs) == 1)
        return lrs[0]
