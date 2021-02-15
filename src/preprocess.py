#!./env python

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
from collections import Counter

import numpy as np

dataset_stats = {'cifar10': {'mean': (0.49139968, 0.48215841, 0.44653091),
                             'std': (0.24703223, 0.24348513, 0.26158784)},
                 'cifar100': {'mean': (0.50707516, 0.48654887, 0.44091784),
                              'std': (0.26733429, 0.25643846, 0.27615047)},
                 'mnist': {'mean': (0.1306605,),
                           'std': (0.3081078,)},
                 }


## Custom dataset with per-sample weight
# from torch.utils.data import Dataset, DataLoader
class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, weights=dict(), config=None):
        assert(isinstance(dataset, torch.utils.data.Dataset))
        for key in weights:
            assert(len(weights[key]) == len(dataset)), (key, len(weights[key]), len(dataset))
        self.dataset = dataset
        self.weights = weights

        # save attributes
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.targets = dataset.targets

        self.config = config
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        # Your transformations here (or set it in CIFAR10)
        # weight = self.weights[index]
        weight = dict([(key, self.weights[key][index]) for key in self.weights])
        weight['index'] = index

        # sanity check
        if 'alpha' in weight and weight['alpha'] < 0.2:
            if 'reg' in weight:
                assert(weight['reg'] == 0), 'adding label smoothing to clean loss will cause false robustness! Index: %i, ls: %.2f, alpha: %.2f' % (index, weight['reg'], weight['alpha'])
            else:
                assert(not hasattr(self.config, 'label_smoothing') or self.config.label_smoothing == 0), 'adding label smoothing to clean loss will cause false robustness! Index: %i, ls: %.2f, alpha: %.2f' % (index, self.config.label_smoothing, weight['alpha'])
        
        return data, target, weight

    def __len__(self):
        return len(self.dataset)


class Loaders:
    pass

def summary(dataset, classes, class_to_idx):
    print('shape: ', dataset[0][0].size())
    print('size: %i' % len(dataset))
    print('num classes: %i' % len(classes))
    print('---------------------------')
    if len(dataset[0]) == 2:
        d = dict(Counter([classes[label] for _, label in dataset]).most_common())
    else:
        d = dict(Counter([classes[label] for _, label, _ in dataset]).most_common())
    for c in classes:
        if c in d:
            print('%s: %i' % (c, d[c]))
        else:
            print('%s: %i' % (c, 0))
    print('\n')


def get_loaders(dataset='cifar10', classes=None, batch_size=128,
                shuffle_train_loader=True, random_augment=True,
                trainsize=None, testsize=None, 
                trainsubids=None, testsubids=None, # for select ids
                trainextrasubids=[],
                weights={},
                data_dir='/home/jingbo/chengyu/Initialization/data',
                n_workers=4, download=False, config=None):
    """
        Property: support selection of classes
    """
    # TODO: Infer trainsize and testsize

    if dataset == 'mnist':
        dataloader = datasets.MNIST
        # targets in mnist are torch tensors
        to_idx = lambda x: x.item()
    elif dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        # targets in cifar are integers
        to_idx = lambda x: x
    elif dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        to_idx = lambda x: x
    else:
        raise KeyError('dataset: %s ' % dataset)

    # classes = ('dog', 'cat') # , 'bird', 'frog', 'horse')

    # ----------
    # get dataset
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    if dataset == 'mnist':
        transform_train = transforms.Compose([
                transforms.ToTensor(), # To tensor implicitly applies a min-max scaler, such that the range is [0, 1]
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])

        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])
    elif dataset in ['cifar10', 'cifar100']:
        if random_augment:
            transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_stats[dataset]['mean'],
                                         dataset_stats[dataset]['std']),])
        else:
            transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_stats[dataset]['mean'],
                                         dataset_stats[dataset]['std']),])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])
    else:
        raise KeyError('dataset: %s ' % dataset)


    trainset = dataloader(root=data_dir,
                          train=True,
                          download=download,
                          transform=transform_train)
    # if weights: # len(weights) > 0:
    trainset = WeightedDataset(trainset, weights=weights, config=config)

    testset = dataloader(root=data_dir,
                         train=False,
                         download=download,
                         transform=transform_test)

    # just add the index, no additional weights allowed
    testset = WeightedDataset(testset)

    # ---------
    # select sub-classes
    if not classes:
        # save attributes
        classes = trainset.classes
        class_to_idx = trainset.class_to_idx

        # Extra subsets to evaluate training set
        trainextrasubsets = []
        if trainextrasubids:
            # assert(trainsubids is not None), 'Eval subsets are set, but train subset is not. Double check!'
            # sanity check
            if trainsubids is not None:
                trainids_ = trainsubids
            else:
                trainids_ = np.arange(len(trainset))
            for ids in trainextrasubids:
                assert(np.all(np.isin(ids, trainids_))), 'Eval subset ids is not included in the trainset ids ! Double check!'

            # select extra subids before changing trainset
            trainextrasubsets = [torch.utils.data.Subset(trainset, ids) for ids in trainextrasubids]

        trainsubids_ = None

        # Randomly select subset
        if trainsize is not None:
            assert trainsubids is None, 'selected based on ids is prohibited when size is enabled'
            assert trainsize < len(trainset), 'training set has only %i examples' % len(trainset)
            # random select subsets
            # trainids = np.arange(len(trainset))
            # trainsubids = np.random.sample(trainids, trainsize)
            trainsubids_ = np.random.choice(len(trainset), trainsize,
                                            replace=False)
            targets = [trainset.targets[i] for i in trainsubids_]
            trainset = torch.utils.data.Subset(trainset, trainsubids_)

            # recover attributes
            trainset.classes = classes
            trainset.class_to_idx = class_to_idx
            trainset.targets = targets

            # # Select a subset of training set in this training subset only for robustness validation
            # if len(testset) < len(trainset):
            #     subids = random.sample(range(len(trainset)), len(testset))
            #     trainextrasubsets.append(torch.utils.data.Subset(trainset, subids))
            # else:
            #     trainextrasubsets.append(trainset)

        # Specified subset
        if trainsubids is not None:
            assert(isinstance(trainsubids, np.ndarray))
            # targets = np.array(trainset.targets)[trainsubids].tolist()
            targets = [trainset.targets[i] for i in trainsubids]
            trainset = torch.utils.data.Subset(trainset, trainsubids)

            # recover attributes
            trainset.classes = classes
            trainset.class_to_idx = class_to_idx
            trainset.targets = targets

            # to be consistent with the variable used above
            trainsubids_ = np.array(trainsubids)

            # # Select a subset of training set in this training subset only for robustness validation
            # if len(testset) < len(trainset):
            #     subids = random.sample(range(len(trainset)), len(testset))
            #     trainextrasubsets.append(torch.utils.data.Subset(trainset, subids))
            # else:
            #     trainextrasubsets.append(trainset)

        if testsize is not None or testsubids is not None:
            raise NotImplementedError
    else:
        assert isinstance(classes, tuple) or isinstance(classes, list)
        assert all([c in trainset.classes for c in classes]), (trainset.classes, classes)

        idx_classes = [trainset.class_to_idx[c] for c in classes]
        idx_convert = dict([(idx, i) for i, idx in enumerate(idx_classes)])
        class_to_idx = dict([(c, i) for i, c in enumerate(classes)])

        # select in-class indices
        trainids = [i for i in range(len(trainset)) if trainset.targets[i] in idx_classes]
        testids = [i for i in range(len(testset)) if testset.targets[i] in idx_classes]

        # modify labels. 0-9(10 classes) -> 0-1(2 classes)
        for idx in trainids:
            trainset.targets[idx] = idx_convert[to_idx(trainset.targets[idx])]
        for idx in testids:
            testset.targets[idx] = idx_convert[to_idx(testset.targets[idx])]

        # select a subset if needed
        if trainsize:
            random.shuffle(trainids)
            trainids = trainids[:trainsize]
        if testsize:
            random.shuffle(testids)
            testids = testids[:testsize]
        if trainsubids is not None or testsubids is not None:
            raise NotImplementedError
        
        trainset = torch.utils.data.Subset(trainset, trainids)
        testset = torch.utils.data.Subset(testset, testids)

    if config.traintest:
        # select a subset of training set for robustness validation
        # ~~Do it before sampling training set so that the training accuracy is always comparable~~ ??
        if len(testset) < len(trainset):
            subids = random.sample(range(len(trainset)), len(testset))
            trainsubset = torch.utils.data.Subset(trainset, subids)
        else:
            trainsubset = trainset

    print('- training set -')
    summary(trainset, classes, class_to_idx)
    print('- test set -')
    summary(testset, classes, class_to_idx)
    if config.traintest:
        print('- training sub set -')
        summary(trainsubset, classes, class_to_idx)
    if trainextrasubsets:
        for subset in trainextrasubsets:
            print('- extra training sub set for evaulation -')
            summary(subset, classes, class_to_idx)

    # print(len(trainset))
    # print(len(testset))

    # ----------
    # deploy loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=shuffle_train_loader, num_workers=n_workers)
    if config.traintest:
        traintestloader = torch.utils.data.DataLoader(trainsubset, batch_size=batch_size,
                                                      shuffle=False, num_workers=n_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=n_workers)


    # integrate
    n_channel = trainset[0][0].size()[0]
    shape = trainset[0][0].size()

    loaders = Loaders()
    loaders.trainloader = trainloader
    loaders.testloader = testloader
    if config.traintest:
        loaders.traintestloader = traintestloader
    loaders.classes = classes
    loaders.class_to_idx = class_to_idx
    loaders.num_classes = len(classes)
    loaders.n_channel = n_channel
    loaders.shape = shape
    loaders.trainset = trainset
    loaders.testset = testset
    if config.traintest:
        loaders.trainsubset = trainsubset
    loaders.trainids = trainsubids_

    if trainextrasubsets:
        loaders.trainextrasubsets = trainextrasubsets
        loaders.trainextraloaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                                                 shuffle=False, num_workers=n_workers) for subset in trainextrasubsets]

    # print(classes)
    return loaders

if __name__ == '__main__':
    loaders = get_loaders() # classes=('dog', 'cat'))
    print(loaders.classes)
