#!./env python

from .autoattack import AutoAttack
import torch
import numpy as np
import os

import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from ..preprocess import dataset_stats

class AAAttacker:
    """ 
        wrapping class for autoattack setup
            default setting aligned with example in auto attack repo
    """
    def __init__(self, net, eps=8.,
                 normalize=True,
                 mode='standard',
                 sample=10000,
                 rand_sample=False,
                 seed=None,
                 path='.',
                 log_path=None,
                 dataset='cifar10', batch_size=128, device=None,
                 data_dir='./data'):

        self.net = net
        self.eps = eps / 255.
        self.normalize = normalize

        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        if data_dir is not None:
            print('>>>>>>>>>>> get dataset (w/o normalization)..')
            self.__set_dataset(data_dir)

            print('>>>>>>>>>>> set evaluation subset..')
            self.__set_idx(sample, rand_sample, seed)

            print('>>>>>>>>>>> set evaluating instance..')
            if log_path is None:
                log_path = '%s/log_certify' % path
                if sample < len(self.testset):
                    log_path += '_%i' % sample
                if rand_sample:
                    log_path += '_rand'
                if mode != 'standard':
                    log_path += '_fast'
                log_path += '.txt'
            else:
                log_path = os.path.join(path, log_path)
        else:
            # used for evaluating on custom data 
            log_path = '%s/log_tmp.txt' % path

        self.ids_save_path = os.path.splitext(log_path)[0] + '_correct_ids.npy'

        if mode == 'standard':
            version = 'standard'
            attacks_to_run = []
        elif mode == 'fast':
            version = 'custom'
            attacks_to_run = ['apgd-ce', 'apgd-t']
        else:
            raise KeyError(mode)
        self.adversary = AutoAttack(self.get_logits,
                                    norm='Linf',
                                    eps=self.eps,
                                    log_path=log_path,
                                    version=version,
                                    attacks_to_run=attacks_to_run,
                                    device=device)

    def evaluate(self, mode='test', x_test=None, y_test=None):
        if x_test is None:
            assert(y_test is None)
            assert(hasattr(self, 'get_idx')), 'default testset is not set!'
            idx = self.get_idx(mode)
            dataset = self.datasets[mode]
            # x_test = torch.cat([self.testset[i][0].unsqueeze(0) for i in idx])
            # y_test = torch.Tensor([self.testset[i][1] for i in idx]).long()
            x_test = torch.cat([dataset[i][0].unsqueeze(0) for i in idx])
            y_test = torch.Tensor([dataset[i][1] for i in idx]).long()

        x_ad, acc, flags = self.adversary.run_standard_evaluation(x_test,
                                                                  y_test,
                                                                  bs=self.batch_size)
        ids = flags.nonzero().squeeze().cpu().numpy()
        # if x_test is None:
        #     ids = indices[ids]
        with open(self.ids_save_path, 'wb') as f:
            np.save(f, ids)
        return x_ad, acc * 100.

    def get_logits(self, inputs):
        # logit function that preceded by input normalization
        if self.normalize:
            inputs = self._normalize(inputs)
        return self.net(inputs)

    def _normalize(self, X):
        X = X.to(self.device)
        mu = torch.tensor(dataset_stats[self.dataset]['mean']).view(3, 1, 1).to(self.device)
        std = torch.tensor(dataset_stats[self.dataset]['std']).view(3, 1, 1).to(self.device)
        return (X - mu) / std

    def __set_dataset(self, data_dir):
        # auto attack assume a [0,1] input
        # therefore need to rebuild the dataset here
        transform_list = [transforms.ToTensor()]
        transform_chain = transforms.Compose(transform_list)
        self.testset = datasets.CIFAR10(root=data_dir,
                                        train=False,
                                        transform=transform_chain,
                                        download=False)

        self.trainset = datasets.CIFAR10(root=data_dir,
                                         train=True,
                                         transform=transform_chain,
                                         download=False)

        self.datasets = {'test': self.testset,
                         'train': self.trainset}

    def __set_idx(self, sample, rand_sample, seed):
        if sample is None or sample >= len(self.testset):
            print('>>>>>>>>>>> evaluate on the entire testset or training set..')
            def get_idx(mode='test'):
                if mode == 'test':
                    return np.arange(len(self.testset))
                if mode == 'train':
                    return np.arange(len(self.trainset))
                raise KeyError(mode)
            self.get_idx = get_idx
            # self.get_idx = lambda: np.arange(len(self.testset))
            return

        if not rand_sample:
            print('>>>>>>>>>>> get fixed evaluation indices..')
            if seed is not None:
                np.random.seed(seed)
                self.test_idx = self.__rand_idx(sample)
                self.train_idx = self.__rand_idx(sample, len(self.trainset))
            else:
                self.test_idx = np.arange(sample)
                self.train_idx = np.arange(sample)
            # self.get_idx = lambda: self.idx
            def get_idx(mode='test'):
                if mode == 'test':
                    return self.test_idx
                if mode == 'train':
                    return self.train_idx
                raise KeyError(mode)
            self.get_idx = get_idx
            return

        print('>>>>>>>>>>> get random evaluation indices during training..')
        def get_idx(mode='test'):
            if mode == 'test':
                return self.__rand_idx(sample)
            if mode == 'train':
                return self.__rand_idx(sample, len(self.trainset))
            raise KeyError(mode)
        self.get_idx = get_idx
        # self.get_idx = lambda: self.__rand_idx(sample)

    def __rand_idx(self, sample, length=None):
        if length is None:
            length = len(self.testset)
        return np.random.choice(range(length),
                                sample,
                                replace=False)

