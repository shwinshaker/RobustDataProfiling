#!./env python

import torch
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

    time_start = time.time()
    tester = Tester(loaders, net, optimizer, config, time_start)
    adtrainer = AdTrainer(loaders, net, optimizer=optimizer, criterion=criterion, config=config, time_start=time_start)

    for epoch in range(config.epoch_start, config.epochs):        
        net.train()
        for i, (inputs, labels, weights) in enumerate(loaders.trainloader, 0):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            loss = adtrainer._loss(inputs, labels, weights, epoch=epoch)

            # update net
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test mode
        net.eval() 
        tester.update(epoch, i)
        adtrainer.update(epoch, i)
        adtrainer.reset(epoch)
        if scheduler:
            scheduler.step()

        # save model
        save_checkpoint(epoch, net, optimizer, scheduler)
        if config.save_interval:
            # save model sequentially
            if epoch % config.save_interval == 0:
                torch.save(net.state_dict(), 'model-%s.pt' % epoch)

    torch.save(net.state_dict(), 'model.pt')

    tester.close()
    adtrainer.close()
