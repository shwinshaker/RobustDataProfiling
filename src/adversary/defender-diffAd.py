#!./env python

import torch
import torch.nn as nn
from ..utils import Logger, AverageMeter, accuracy
from ..utils import ExampleTracker
from . import attack, scale_step
from .loss import trades_loss, llr_loss, mart_loss, bce_loss, fat_loss, gairat_loss
import time
import copy

__all__ = ['AdTrainer']

class AdTrainer:
    def __init__(self, loaders, net, optimizer, criterion=None, config=None, time_start=None):
        self.loaders = loaders
        self.net = net
        self.optimizer = optimizer # only used for getting current learning rate
        self.criterion = criterion
        self.config = config
        self.device = self.config.device
        self.time_start = time_start

        # differential attack
        self.ad_net = copy.deepcopy(self.net)
        self.ad_net.eval()

        # target
        self.target = None
        if config.target is not None:
            self.target = loaders.class_to_idx[config.target]

        # scale epsilon (each channel is different because different range)
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

        if hasattr(self, 'extra_metrics'):
            # log for 'false' training stats
            time_elapse = (time.time() - self.time_start)/60
            logs_base = [epoch, i, self.__get_lr(), time_elapse]
            logs_e = [_ for _ in logs_base]
            logs_e.extend([self.meters[m].avg for m in self.extra_metrics]) # keep the order
            self.logger_e.append(logs_e)

        ## In-situ masked subset
        if self.config.exTrack:
            # make a record
            self.exLog.step(epoch)

        ## epoch is current epoch + 1
        self.epoch = epoch + 1

        ## update attack model
        print('attack model updated')
        self.ad_net = copy.deepcopy(self.net)
        self.ad_net.eval()

    def reset(self, epoch):
        assert(epoch == self.epoch - 1), 'reset is not called after update!'

        ## reset some logger
        if hasattr(self, 'extra_metrics'):
            # reset meters
            for m in self.meters:
                self.meters[m].reset()

        if self.config.exTrack:
            self.exLog.reset()

    def __ad_setup(self):

        base_names = ['Epoch', 'Mini-batch', 'lr', 'Time-elapse(Min)']
        self.logger_e = Logger('log_extra.txt', title='log for deprecated metrics', resume=self.config.resume)

        if not self.config.adversary:
            self._loss = self._clean_loss

            # log for sample robust correctness
            if self.config.exTrack:
                self.exLog = ExampleTracker(self.loaders)

            # log for 'false' training loss and acc, aligned with previous work
            self.extra_metrics = ['Train-Loss', 'Train-Acc']
            self.logger_e.set_names(base_names + self.extra_metrics)
            self.meters = dict([(m, AverageMeter()) for m in self.extra_metrics])
            return


        if self.config.adversary in ['gaussian', 'fgsm', 'pgd', 'fgsm_manifold', 'pgd_manifold']:
            self._loss = self._ad_loss

            # log for sample robust correctness
            if self.config.exTrack:
                self.exLog = ExampleTracker(self.loaders)

            # log for 'false' training loss and acc, aligned with previous work
            self.extra_metrics = ['Train-Loss', 'Train-Acc', 'Train-Loss-Ad', 'Train-Acc-Ad']
            self.logger_e.set_names(base_names + self.extra_metrics)
            self.meters = dict([(m, AverageMeter()) for m in self.extra_metrics])
            return

        # No extra logger: 'false' training accuracy not recorded for integrated loss - needs to code within specific loss function
        # other things not supported currectly..
        if self.config.target:
            raise NotImplementedError('Targeted attack not supported! TODO..')
        # if hasattr(self.config, 'alpha_sample_path') and self.config.alpha_sample_path:
        #     # for trades, this was implemented, but try incorporate into the loss function
        #     raise NotImplementedError('Sample-wise trading not supported! TODO..')
        if hasattr(self.config, 'reg_sample_path') and self.config.reg_sample_path:
            raise NotImplementedError('Sample-wise regularization Not supported!')

        if self.config.adversary == 'trades':
            self._loss = self._trades_loss
        elif self.config.adversary == 'mart':
            self._loss = self._mart_loss
        elif self.config.adversary == 'fat':
            self._loss = self._fat_loss
        elif self.config.adversary == 'gairat':
            self._loss = self._gairat_loss
        else:
            raise NotImplementedError(self.config.adversary)

        if self.config.exTrack:
            self.exLog = ExampleTracker(self.loaders)

        self.extra_metrics = ['Train-Loss-Ad', 'Train-Acc-Ad']
        self.logger_e.set_names(base_names + self.extra_metrics)
        self.meters = dict([(m, AverageMeter()) for m in self.extra_metrics])
        return

        if self.config.exTrack:
            raise NotImplementedError('Example tracking not supported! TODO..')

        if self.config.adversary == 'llr':
            self._loss = self._llr_loss
            return

        raise KeyError('Unexpected adversary %s' % self.config.adversary)

    def _clean_loss(self, inputs, labels, weights, epoch=None):
        outputs = self.net(inputs)
        if 'reg' in weights:
            loss = self.criterion(outputs,
                                  labels,
                                  weights=weights['reg'].to(self.device))
        else:
            loss = self.criterion(outputs, labels)
        prec1, = accuracy(outputs.data, labels.data)
    
        # keep for record
        self.meters['Train-Loss'].update(loss.mean().item(), inputs.size(0))
        self.meters['Train-Acc'].update(prec1.item(), inputs.size(0))  # accuracy on clean data
        if self.config.exTrack:
            self.exLog.update(outputs, labels, weights['index'].to(self.device), epoch=self.epoch)
        return loss.mean()

    def _ad_loss(self, inputs, labels, weights, epoch=None):

        # -------- clean loss
        loss = 0.
        # if pure ad loss and sample-wise alpha not enabled, don't have to do this part
        if self.config.alpha < 1.0 or 'alpha' in weights:
            loss = self._clean_loss(inputs, labels, weights)

        # ------- ad loss
        eps_weight = None
        if 'weps' in weights:
            eps_weight = weights['weps']

        self.net.eval()
        ctr = nn.CrossEntropyLoss() # Don't change the criterion in adversary generation part -- maybe change it later
        # inputs_ad = attack(self.net, ctr, inputs, labels, weight=eps_weight,
        inputs_ad = attack(self.ad_net, ctr, inputs, labels, weight=eps_weight,
                           adversary=self.config.adversary,
                           eps=self.config.eps,
                           pgd_alpha=self.config.pgd_alpha,
                           pgd_iter=self.config.pgd_iter,
                           randomize=self.config.rand_init,
                           target=self.target,
                           config=self.config)
        self.net.train()
        outputs_ad = self.net(inputs_ad)

        if 'reg' in weights:
            loss_ad = self.criterion(outputs_ad,
                                     labels,
                                     weights=weights['reg'].to(self.device))
        else:
            loss_ad = self.criterion(outputs_ad, labels)
            # print('bce!')
            # loss_ad = bce_loss(outputs_ad, labels, reduction='none')

        # -------- combine two loss
        if 'alpha' in weights:
            # sample-wise weighting
            assert(loss.size(0) == inputs.size(0)), (loss.size(0), inputs.size(0))
            alpha = weights['alpha'].to(self.device)
            assert(loss.size() == loss_ad.size() == alpha.size()), (loss.size(), loss_ad.size(), alpha.size())
        else:
            alpha = self.config.alpha

        if 'lambda' in weights:
            lmbd = weights['lambda'].to(self.device)
        else:
            lmbd = torch.ones(inputs.size(0)).to(self.device)

        assert(loss_ad.size() == lmbd.size()), (loss_ad.size(), lmbd.size())
        loss *= (1 - alpha)
        loss += alpha * loss_ad * lmbd / lmbd.sum()
        loss = loss.sum()

        # -------- recording
        prec1_ad, = accuracy(outputs_ad.data, labels.data)
        self.meters['Train-Loss-Ad'].update(loss_ad.mean().item(), inputs.size(0))
        self.meters['Train-Acc-Ad'].update(prec1_ad.item(), inputs.size(0))
        if self.config.exTrack:
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss

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

    def _trades_loss(self, inputs, labels, weights, epoch=None):
        # note: The official implementation use CE + KL * beta - amounts to alpha~= 0.85
        #       Previously we use (1-alpha) * CE + alpha * KL
        # integrate clean loss in trades loss
        # sample-weighting in trades loss - later
        loss, outputs_ad = trades_loss(self.net, inputs, labels, weights,
                                      eps=self.config.eps,
                                      alpha=self.config.pgd_alpha,
                                      num_iter=self.config.pgd_iter,
                                      norm='linf',
                                      rand_init=self.config.rand_init,
                                      config=self.config)

        # -------- recording
        prec1_ad, = accuracy(outputs_ad.data, labels.data)
        self.meters['Train-Loss-Ad'].update(loss.item(), inputs.size(0))
        self.meters['Train-Acc-Ad'].update(prec1_ad.item(), inputs.size(0))
        if self.config.exTrack:
            # raise NotImplementedError('Generate ad examples using PGD, otherwise not fair!')
            outputs_ad = self.__get_pgd_ad(inputs, labels)
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss

    def _mart_loss(self, inputs, labels, weights, epoch=None):
        loss, outputs_ad = mart_loss(self.net, inputs, labels, weights,
                                     eps=self.config.eps,
                                     alpha=self.config.pgd_alpha,
                                     num_iter=self.config.pgd_iter,
                                     norm='linf',
                                     rand_init=self.config.rand_init,
                                     config=self.config)

        # -------- recording
        prec1_ad, = accuracy(outputs_ad.data, labels.data)
        self.meters['Train-Loss-Ad'].update(loss.item(), inputs.size(0))
        self.meters['Train-Acc-Ad'].update(prec1_ad.item(), inputs.size(0))
        if self.config.exTrack:
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss

    def __get_tau(self, epoch):
        tau = self.config.fat_taus[0]
        if epoch > self.config.fat_milestones[0]:
            tau = self.config.fat_taus[1]
        if epoch > self.config.fat_milestones[1]:
            tau = self.config.fat_taus[2]
        return tau

    def _fat_loss(self, inputs, labels, weights, epoch=None):
        tau = self.__get_tau(epoch)
        loss, outputs_ad, ad_steps = fat_loss(self.net, inputs, labels, weights,
                                              eps=self.config.eps,
                                              alpha=self.config.pgd_alpha,
                                              num_iter=self.config.pgd_iter,
                                              tau=tau,
                                              norm='linf',
                                              rand_init=self.config.rand_init,
                                              config=self.config)

        prec1_ad, = accuracy(outputs_ad.data, labels.data)
        self.meters['Train-Loss-Ad'].update(loss.item(), inputs.size(0))
        self.meters['Train-Acc-Ad'].update(prec1_ad.item(), inputs.size(0))
        if self.config.exTrack:
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss

    def _gairat_loss(self, inputs, labels, weights, epoch=None):
        tau = self.__get_tau(epoch)
        loss, outputs_ad, ad_steps = gairat_loss(self.net, inputs, labels, weights,
                                                 eps=self.config.eps,
                                                 alpha=self.config.pgd_alpha,
                                                 num_iter=self.config.pgd_iter,
                                                 tau=tau,
                                                 norm='linf',
                                                 rand_init=self.config.rand_init,
                                                 config=self.config)

        prec1_ad, = accuracy(outputs_ad.data, labels.data)
        self.meters['Train-Loss-Ad'].update(loss.item(), inputs.size(0))
        self.meters['Train-Acc-Ad'].update(prec1_ad.item(), inputs.size(0))
        if self.config.exTrack:
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch, ad_steps=ad_steps)

        return loss

    def _llr_loss(self, inputs, labels, weights, epoch=None):
        loss = llr_loss(self.net, inputs, labels,
                        eps=self.config.eps,
                        alpha=self.config.pgd_alpha,
                        num_iter=self.config.pgd_iter,
                        norm='linf',
                        rand_init=self.config.rand_init,
                        config=self.config)
        return loss

    def __get_lr(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        assert(len(lrs) == 1)
        return lrs[0]

    def close(self):
        if hasattr(self, 'logger_e'):
            self.logger_e.close()
