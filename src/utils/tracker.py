#!./env python

import torch
import torch.nn as nn
import numpy as np
import os

__all__ = ['ExampleTracker']

class ExampleTracker:
    """
        Track learning stats of training examples

            count_wrong: # epochs that an example is misclassified.
                            An array with length N where N is the number of training examples
                            The order corresponds to the order in the CIFAR-10 training set loaded from Pytorch

            epoch_first: epoch that an example is first learned
                            An array with length N where N is the number of training examples
                            The order corresponds to the order in the CIFAR-10 training set loaded from Pytorch

            record_correct: record all events that an example is correctly classified (1), or misclassifed (0)
                                A matrix with length T X N, where T is the number of epochs, N is the number of training examples
                                The order in the column corresponds to the order in the CIFAR-10 training set loaded from Pytorch
                                The order in the row corresponds to the sequential order of training epochs
    """

    __trainsize = 50000

    def __init__(self, loaders, resume=False):
        self.indices = []

        # for sanity check: only allow outputs when all the examples are scanned
        self.count = 0

        if resume:
            self.trainsize = self.__trainsize
            self.count_wrong = np.load('count_wrong.npy')
            self.epoch_first = np.load('epoch_first.npy')
            self.record_correct = np.zeros(self.__trainsize).astype(np.int8) # save some space

            if loaders.trainids is not None:
                self.trainsubids = loaders.trainids
                self.trainsize = len(self.trainsubids)
                # mask out
                ids_ = np.setdiff1d(np.arange(self.__trainsize), self.trainsubids)
                assert(np.all(self.count_wrong[ids_] == -1)), 'masked id mismatch in saved count_wrong'
                assert(np.all(self.epoch_first[ids_] == -2)), 'masked id mismatch in saved epoch_first'
                self.record_correct[ids_] = -1

        else:
            self.trainsize = self.__trainsize
            self.count_wrong = np.zeros(self.__trainsize) # count the number of epochs that an example is correct during training
            self.epoch_first = -np.ones(self.__trainsize)
            self.record_correct = np.zeros(self.__trainsize).astype(np.int8) # save some space

            if loaders.trainids is not None:
                self.trainsubids = loaders.trainids
                self.trainsize = len(self.trainsubids)
                # mask out
                ids_ = np.setdiff1d(np.arange(self.__trainsize), self.trainsubids)
                self.count_wrong[ids_] = -1
                self.epoch_first[ids_] = -2
                self.record_correct[ids_] = -1

    def get_indices(self):
        assert(self.count == self.trainsize), 'num of ex scanned are short! Current: %i Expected: %i' % (self.count, self.trainsize)
        return np.hstack(self.indices)

    def update(self, outputs, labels, ids, epoch=None):
        # sanity check
        if hasattr(self, 'trainsubids'):
            assert(np.all(np.in1d(ids.cpu().numpy(), self.trainsubids)))

        _, preds = outputs.topk(1, 1, True, True)
        correct_ids = ids[preds.squeeze().eq(labels)].cpu().numpy()
        wrong_ids = ids[~preds.squeeze().eq(labels)].cpu().numpy()

        # For early stopping usage
        self.indices.append(wrong_ids)
        self.count += outputs.size(0) # For sanity check
        
        # Record number of wrong epochs throughout the training
        self.count_wrong[wrong_ids] += 1

        # Record the epoch that an example is first learned
        for i in correct_ids:
            if self.epoch_first[i] == -1:
                self.epoch_first[i] = epoch

        # Record all correct learning events in this epoch
        self.record_correct[correct_ids] = 1
        self.record_correct[wrong_ids] = 0

    def step(self, epoch):
        # For log
        print('[%i] %i out of %i examples are wrong' % (epoch, len(self.get_indices()), self.trainsize))
        print('[%i] max count: %i, min_count: %i' % (epoch, np.max(self.count_wrong), np.min(self.count_wrong)))

        # Save
        with open('./count_wrong.npy', 'wb') as f:
            np.save(f, self.count_wrong, allow_pickle=True)
        with open('./epoch_first.npy', 'wb') as f:
            np.save(f, self.epoch_first, allow_pickle=True)

        # Save all learning events for future processing
        self.__push(self.record_correct)

    def __push(self, array, file_name='record_correct.npy'):
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                record = np.load(f, allow_pickle=True)
            record = np.vstack([record, array])
        else:
            record = np.array(array)
        with open(file_name, 'wb') as f:
            np.save(f, record, allow_pickle=True)

    def reset(self):
        # Clear
        self.indices = []
        self.count = 0
