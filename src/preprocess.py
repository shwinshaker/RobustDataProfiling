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

## Custom dataset with index recorded
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
        weight = dict([(key, self.weights[key][index]) for key in self.weights])
        weight['index'] = index

        return data, target, weight

    def __len__(self):
        return len(self.dataset)


def summary(dataset, classes, class_to_idx):
    ## print loader info
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


def get_loaders(dataset='cifar10', batch_size=128,
                shuffle_train_loader=True,
                trainsize=None, testsize=None, 
                trainsubids=None, testsubids=None,
                weights={},
                data_dir='/home/jingbo/chengyu/Initialization/data',
                n_workers=4, download=False, config=None):

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

    # ----------
    # get dataset
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    if dataset == 'mnist':
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])

        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])
    elif dataset in ['cifar10', 'cifar100']:
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
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
    trainset = WeightedDataset(trainset, weights=weights, config=config)

    testset = dataloader(root=data_dir,
                         train=False,
                         download=download,
                         transform=transform_test)
    testset = WeightedDataset(testset)

    # ---------
    classes = trainset.classes
    class_to_idx = trainset.class_to_idx

    trainsubids_ = None
    # Randomly select subset
    if trainsize is not None:
        assert trainsubids is None, 'selected based on ids is prohibited when size is enabled'
        assert trainsize < len(trainset), 'training set has only %i examples' % len(trainset)
        # random select subsets
        trainsubids_ = np.random.choice(len(trainset), trainsize,
                                        replace=False)
        targets = [trainset.targets[i] for i in trainsubids_]
        trainset = torch.utils.data.Subset(trainset, trainsubids_)

        # recover attributes
        trainset.classes = classes
        trainset.class_to_idx = class_to_idx
        trainset.targets = targets

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

    if testsize is not None or testsubids is not None:
        raise NotImplementedError

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

    class Loaders:
        pass
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

    return loaders

if __name__ == '__main__':
    loaders = get_loaders() # classes=('dog', 'cat'))
    print(loaders.classes)
