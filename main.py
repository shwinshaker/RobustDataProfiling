#!./env python

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from copy import deepcopy
import random
import time
import datetime
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from src.models import get_vgg
import src.models as models
from src.pipeline import train
from src.preprocess import get_loaders
from src.utils import Dict2Obj

from robustCertify import robust_certify

def train_wrap(**config):
    config = Dict2Obj(config)

    start = time.time()

    # time for log
    print('\n=====> Current time..')
    print(datetime.datetime.now())

    # environment set
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Random seed
    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.manualSeed)

    ## ------------------------------- data ----------------------------------
    print('=====> Loading data..')
    trainsubids = None
    if hasattr(config, 'train_subset_path') and config.train_subset_path:
        with open(config.train_subset_path, 'rb') as f:
            trainsubids = np.load(f)

    loaders = get_loaders(dataset=config.dataset, batch_size=config.batch_size,
                          trainsize=config.trainsize, testsize=config.testsize,
                          trainsubids=trainsubids,
                          data_dir=config.data_dir,
                          config=config)

    ## --------------------------------- criterion ------------------------------- 
    config.reduction = 'none' # Always prevent reduction, do reduction in training script
    criterion = nn.CrossEntropyLoss(reduction=config.reduction)


    ## ---------------------------------  model ------------------------------- 
    print('=====> Initializing model..')
    if 'vgg' in config.model:
        net = get_vgg(model=config.model, batch_norm=config.bn, num_classes=loaders.num_classes, n_channel=loaders.n_channel, gain=config.gain, dataset=config.dataset).to(config.device)
    elif config.model in ['PreActResNet18']:
        net = models.__dict__[config.model]().to(config.device)
    elif 'wrn' in config.model:
        net = models.__dict__[config.model](depth=config.depth, widen_factor=config.width, num_classes=loaders.num_classes, n_channel=loaders.n_channel).to(config.device)
    else:
        raise KeyError(config.model)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    print(net)
    print("     Total params: %.2fM" % (sum(p.numel() for p in net.parameters())/1000000.0))

    ## -- Load weights
    if config.state_path:
        print('=====> Loading pre-trained weights..')
        assert(not config.resume), 'pre-trained weights will be overriden by resume checkpoint! Resolve this later!'
        state_dict = torch.load(config.state_path, map_location=config.device)
        net.load_state_dict(state_dict)

    ## ---------------------------------- optimizer --------------------------------
    print('=====> Initializing optimizer..')
    if config.opt.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=config.lr, weight_decay=config.wd, momentum=config.momentum)
    elif config.opt.lower() == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=config.lr, weight_decay=config.wd)
    elif config.opt.lower() == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=config.lr, weight_decay=config.wd, momentum=config.momentum)
    elif config.opt.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.wd)
    else:
        raise KeyError(config.optimizer)

    ## -------------------------------------  lr scheduler ---------------------------------- 
    print('=====> Initializing scheduler..')
    scheduler = None
    if config.scheduler:
        if config.scheduler == 'multistep':
            if config.milestones:
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
        else:
            raise KeyError(config.scheduler)

    ## -- Load checkpoint when resuming
    config.epoch_start = 0
    if config.resume:
        print('=====> Loading state..')
        checkpoint = torch.load(config.resume_checkpoint, map_location=config.device)
        config.epoch_start = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if config.scheduler and config.milestones:
            # note: Previously scheduler saving has a problem (get a length 2 runningtime error)
            #       try the following instead if loading previous checkpoints
            # scheduler.load_state_dict(checkpoint['scheduler'][0])
            scheduler.load_state_dict(checkpoint['scheduler'])

    ## ------------------------------------- adversary ---------------------------------- 
    print('=====> Training..')
    train(config, net=net, loaders=loaders, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    print('     Finished.. %.3f' % ((time.time() - start) / 60.0))


if __name__ == '__main__':

    # run from json
    with open('para.json') as json_file:
        config = json.load(json_file)
        print(config)
    train_wrap(**config)

    # cerfity using AA
    print('\n> --------------- Start robustness certification using auto attack ----------------')
    model_parallel = False
    if ',' in str(config['gpu_id']):
        model_parallel = True
    robust_certify(model=config['model'], depth=config['depth'], width=config['width'], model_parallel=model_parallel,
                   gpu_id=config['gpu_id'], state='best',
                   mode='fast', data_dir=config['data_dir'])
    robust_certify(model=config['model'], depth=config['depth'], width=config['width'], model_parallel=model_parallel,
                   gpu_id=config['gpu_id'], state='last',
                   mode='fast', data_dir=config['data_dir'])
 
    # clear up temp files
    os.remove('checkpoint.pth.tar')
    if not config['save_model']:
        os.remove('best_model.pt')
        os.remove('model.pt')

    # time for log
    print('\n=====> Current time..')
    print(datetime.datetime.now())


