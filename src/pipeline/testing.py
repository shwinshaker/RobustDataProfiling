#!./env python

import torch
import torch.nn as nn
from ..utils import test, Logger
from ..adversary import ad_test, scale_step
from ..adversary import AAAttacker
import time
import copy

from ..analyses import load_log
import numpy as np
import os

__all__ = ['Tester']

def nan_filter(arr):
    arr = np.array(arr)
    arr[np.isnan(arr)] = 0
    return arr

def get_best_acc(path='.', option='robust'):
    stats = load_log(os.path.join(path, 'log.txt'), window=1)
    if option == 'robust':
        return np.max(nan_filter(stats['Test-Acc-Ad']))
    if option == 'clean':
        return np.max(nan_filter(stats['Test-Acc']))
    raise KeyError(option)

def get_last_time(path='.'):
    return load_log(os.path.join(path, 'log.txt'))['Time-elapse(Min)'][-1]

class Tester:
    def __init__(self, loaders, net, optimizer, config, time_start):

        self.loaders = loaders
        self.net = net
        # self.criterion = criterion
        # Criterion in testing is not allowed to change
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.config = config

        # scale epsilon
        config.eps_test = scale_step(config.eps_test, config.dataset, device=config.device)
        config.pgd_alpha_test = scale_step(config.pgd_alpha_test, config.dataset, device=config.device)
        print('scaled eps [test]:', config.eps_test, config.pgd_alpha_test)

        if config.ad_test == 'aa':
            # use alternative tester
            self.__aa_setup() # auto attack takes additional 8 hours..

        # basic logs 
        base_names = ['Epoch', 'Mini-batch', 'lr', 'Time-elapse(Min)']
        self.logger = Logger('log.txt', title='log', resume=config.resume)
        metrics = ['Train-Loss', 'Test-Loss',
                   'Train-Loss-Ad', 'Test-Loss-Ad',
                   'Train-Acc', 'Test-Acc',
                   'Train-Acc-Ad', 'Test-Acc-Ad']
        self.logger.set_names(base_names + metrics)

        self.time_start = time_start
        self.last_end = 0.
        if config.resume:
            self.last_end = get_last_time() # min

        # save best model
        self.best = config.best
        self.best_acc = 0.
        if config.resume:
            self.best_acc = get_best_acc(option=self.best)
            print('> Best: %.2f' % (self.best_acc))

    def __get_lr(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        assert(len(lrs) == 1)
        return lrs[0]
    
    def __test(self, loader):
        return test(loader, self.net, self.criterion, self.config, classes=self.loaders.classes)

    def __ad_test(self, loader):
        return ad_test(loader, self.net, self.criterion, self.config, classes=self.loaders.classes)

    def __aa_setup(self):
        # evaluate on a random subset fixed through training
        self.aa_attacker = AAAttacker(net=self.net,
                                      normalize=True,
                                      mode='fast',
                                      sample=1000,
                                      rand_sample=False,
                                      seed=7,
                                      log_path=None,
                                      device=self.config.device,
                                      data_dir=self.config.data_dir)

    def __ad_test_aa(self, mode='test'):
        ## dummy output
        return 0., self.aa_attacker.evaluate(mode=mode)[1], dict()

    def __update_best(self, epoch, test_acc, test_acc_ad):
        if self.best.lower() == 'robust':
            acc = test_acc_ad
        else:
            acc = test_acc

        if acc > self.best_acc:
            print('> Best got at epoch %i. Best: %.2f Current: %.2f' % (epoch, acc, self.best_acc))
            self.best_acc = acc
            # if self.save_model:
                # self.best_model = deepcopy(self.net)
                # save to file to reduce memory consumption, but may increase time
            torch.save(self.net.state_dict(), 'best_model.pt')

    def update(self, epoch, i):

        assert(not self.net.training), 'Model is not in evaluation mode, calling from wrong place!'
        """
            The net should stay on test mode when attack
                because it makes no sense to calculate the batch statistics of adversary examples
        """

        # train - test
        train_loss, train_prec1 = 0, 0
        train_loss_ad, train_prec1_ad = 0, 0
        if self.config.traintest:
            train_loss, train_prec1 = self.__test(self.loaders.traintestloader)
            if self.config.ad_test == 'aa':
                train_loss_ad, train_prec1_ad = self.__ad_test_aa(mode='train')
            else:
                train_loss_ad, train_prec1_ad = self.__ad_test(self.loaders.traintestloader)

        # test
        test_loss, test_prec1 = self.__test(self.loaders.testloader)
        if self.config.ad_test == 'aa':
            test_loss_ad, test_prec1_ad = self.__ad_test_aa(mode='test')
        else:
            test_loss_ad, test_prec1_ad = self.__ad_test(self.loaders.testloader)

        # best
        self.__update_best(epoch, test_prec1, test_prec1_ad)

        # logs
        time_elapse = (time.time() - self.time_start)/60 + self.last_end
        logs_base = [epoch, i, self.__get_lr(), time_elapse]
        logs = [_ for _ in logs_base]
        logs += [train_loss, test_loss,
                 train_loss_ad, test_loss_ad,
                 train_prec1, test_prec1,
                 train_prec1_ad, test_prec1_ad]
        self.logger.append(logs)

    def close(self):
        self.logger.close()
