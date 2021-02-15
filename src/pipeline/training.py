#!./env python
# Weighted sample version

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from ..preprocess import get_loaders
from ..utils import ParameterTracker, LipTracker, LrTracker, ResidualTracker, RobustTracker, ManifoldTracker, ExampleTracker
from . import Tester
from ..adversary import AdTrainer
import time
import os

__all__ = ['train']

def save_checkpoint(epoch, net, optimizer, scheduler=None, path='.', filename='checkpoint.pth.tar'):
    filepath = os.path.join(path, filename)
    state = {'epoch': epoch + 1,
             'state_dict': net.state_dict(),
             'optimizer' : optimizer.state_dict(),
            }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    torch.save(state, filepath)

def train(config, net=None, loaders=None, criterion=None, optimizer=None, scheduler=None):

    if config.repeat_eps is not None:
        raise NotImplementedError('Not supported in this training script')
    if config.repeat:
        raise NotImplementedError('Deprecated!')
    if config.mixup_alpha:
        raise NotImplementedError('Not supported in this training script')
    if config.epoch_switch > 0:
        raise NotImplementedError('Not supported in this training script')

    time_start = time.time()

    tester = Tester(loaders, net, optimizer, config, time_start)
    adtrainer = AdTrainer(loaders, net, optimizer=optimizer, criterion=criterion, config=config, time_start=time_start)

    # tracker switchs
    if config.lipTrack:
        lipLog = LipTracker(net, device=config.device)
    if config.paraTrack:
        paraLog = ParameterTracker(net)
    if config.lrTrack and isinstance(optimizer, optim.Adagrad):
        lrLog = LrTracker(net)
    if config.resTrack:
        resLog = ResidualTracker(net, device=config.device)
    if config.rbTrack:
        rbLog = RobustTracker(net, loaders, criterion, config, time_start)
    if config.mrTrack:
        mrLog = ManifoldTracker(net, criterion, config=config)

    if config.lmr > 0:
        raise NotImplementedError('Deprecated')

    for epoch in range(config.epoch_start, config.epochs):        
        net.train()
        for i, (inputs, labels, weights) in enumerate(loaders.trainloader, 0):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            loss = adtrainer._loss(inputs, labels, weights, epoch=epoch)

            # update net
            # net.zero_grad() # should be identical
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test mode
        net.eval() 
        tester.update(epoch, i)

        adtrainer.update(epoch, i)

        ## Pre-determined masked subset
        if hasattr(config, 'epoch_mask') and config.epoch_mask > 0:
            if epoch == config.epoch_mask:
                ## switch loaders
                print('--------- switch loaders -----')
                assert(hasattr(config, 'train_subset_path'))
                with open(config.train_subset_path, 'rb') as f:
                    trainsubids = np.load(f)
                assert(hasattr(config, 'alpha_sample_path2'))
                sample_weights = dict()
                with open(config.alpha_sample_path2, 'rb') as f:
                    sampleweights['alpha'] = np.load(f)
                if hasattr(config, 'reg_sample_path') and config.reg_sample_path:
                    with open(config.alpha_sample_path2, 'rb') as f:
                        sampleweights['reg'] = np.load(f)
                trainextrasubids = []
                loaders = get_loaders(dataset=config.dataset, classes=config.classes, batch_size=config.batch_size,
                                      trainsize=config.trainsize, testsize=config.testsize,
                                      trainsubids=trainsubids,
                                      trainextrasubids=trainextrasubids,
                                      weights=sampleweights,
                                      data_dir=config.data_dir)

        if hasattr(config, 'rb_early_stop') and config.epoch_rb_early_stop > 0:
            # TODO: Support add more masks if exists mask already
            if hasattr(config, 'train_subset_path') and config.train_subset_path:
                raise NotImplementedError('Currently not support mix of pre-determined masking nd in-situ masking')
            if hasattr(config, 'alpha_sample_path') and config.alpha_sample_path:
                raise NotImplementedError('Currently not support mix of pre-determined weighting and in-situ weighting')
            if hasattr(config, 'reg_sample_path') and config.reg_sample_path:
                raise NotImplementedError('Prevent using weighted reg in partial early stop mode!')

            if epoch == config.epoch_rb_early_stop:
                if hasattr(config, 'subset_id_path') and config.subset_id_path:
                    with open(config.subset_id_path, 'rb') as f:
                        ids_mask = np.load(f)
                else:
                    ids_mask = adtrainer.exLog.get_indices()
                # sanity check
                assert(np.all(np.isin(ids_mask, np.arange(len(loaders.trainset))))), 'Not all mask indices are included in the indices!'
                if config.rb_early_stop == 'remove':
                    ids_left = np.setdiff1d(np.arange(len(loaders.trainset)), ids_mask)
                    loaders = get_loaders(dataset=config.dataset, classes=config.classes, batch_size=config.batch_size,
                                          trainsize=config.trainsize, testsize=config.testsize,
                                          trainsubids=ids_left,
                                          data_dir=config.data_dir)
                    if config.exTrack:
                        adtrainer.exLog = ExampleTracker(loaders)
                    config.batch_print = max(1, len(loaders.trainloader)//(config.nlogs//config.epochs))

                elif config.rb_early_stop == 'clean':
                    sampleweights = dict()
                    alphas = np.ones(len(loaders.trainset)) * config.alpha
                    alphas[ids_mask] = 0.
                    sampleweights['alpha'] = alphas
                    loaders = get_loaders(dataset=config.dataset, classes=config.classes, batch_size=config.batch_size,
                                          trainsize=config.trainsize, testsize=config.testsize,
                                          weights=sampleweights,
                                          data_dir=config.data_dir)
                else:
                    raise KeyError('Unexpected early stop mode: %s' % config.rb_early_stop)

        # if loaderswitcher.is_time():
        #     # This has to go before update of wrapper, because exLogs are reset after its update
        #     loaders = loaderswitcher.update(epoch, i)

        adtrainer.reset(epoch)

        # trackers
        if config.paraTrack:
            paraLog.update(net, epoch, i)
        if config.lipTrack:
            lipLog.update(epoch, i)
        if config.lrTrack and isinstance(optimizer, optim.Adagrad):
            lrLog.update(epoch, i, optimizer)
        if config.resTrack:
            resLog.update(epoch, i)
        if config.rbTrack:
            rbLog.update(loaders.testloader, epoch, i)
        if config.mrTrack:
            mrLog.record(epoch, i, config.lmr)

        if scheduler:
            scheduler.step()

        # if config.save_model:
            # for resume
        save_checkpoint(epoch, net, optimizer, scheduler)

        if config.save_interval:
            # save model sequentially
            if epoch % config.save_interval == 0:
                torch.save(net.state_dict(), 'model-%s.pt' % epoch)

    # if config.save_model:
    # save last model
    torch.save(net.state_dict(), 'model.pt')

    tester.close()
    adtrainer.close()
    if config.paraTrack:
        paraLog.close()
    if config.lipTrack:
        lipLog.close()
    if config.lrTrack and isinstance(optimizer, optim.Adagrad):
        lrLog.close()
    if config.resTrack:
        resLog.close()
    if config.rbTrack:
        rbLog.close()
    if config.mrTrack:
        mrLog.close()

