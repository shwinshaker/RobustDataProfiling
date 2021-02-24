#!./env python

import argparse
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
import shutil
import yaml
import json
import torch

from src.utils import check_path
from fractions import Fraction

def check_num(num):
    if type(num) in [float, int]:
        return num
    if isinstance(num, str):
        return float(Fraction(num))
    raise TypeError(num)


def read_config(config_file='config.yaml'):

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # -- hyperparas massage --
    # TODO: tracker not implemented for resnet
    if 'resnet' in config['model']:
        config['paraTrack'] = False
        config['lrTrack'] = False
        config['lipTrack'] = False

    if 'resnet' in config['model']:
        if not config['bn']:
            config['model'] = '%s_fixup' % config['model']

    for key in ['eps', 'eps_test', 'lr', 'wd', 'momentum', 'gamma', 'alpha']:
        if key in config and config[key] is not None:
            config[key] = check_num(config[key])

    if config['state_path']:
        # append absolute path
        config['state_path'] = os.path.join(os.getcwd(), 'checkpoints', config['state_path'])


    # -- checkpoint set --
    config['checkpoint'] = '%s_%s' % (config['opt'], config['model'])
    if 'ffn' in config['model']:
        config['checkpoint'] += '_%i_%i' % (config['depth'], config['width'])
    if 'resnet' in config['model']:
        config['checkpoint'] += '%i' % config['depth']
        if config['width'] != 16:
            config['checkpoint'] += '_width=%i' % config['width']
    elif config['model'] in ['ResNet18', 'PreActResNet18', 'FixupPreActResNet18', 'PreActResNetGN18']:
        pass
    elif config['model'] == 'wrn':
        config['checkpoint'] += '-%i-%i' % (config['depth'], config['width'])
    else:
        if config['bn']:
            config['checkpoint'] += '_bn'

    if config['dataset'] != 'cifar10':
        config['checkpoint'] = config['dataset'] + '_' + config['checkpoint']
    if config['adversary']:
        config['checkpoint'] += '_ad'
        config['checkpoint'] += '_%s' % config['adversary']
        if config['adversary'] in ['pgd', 'trades']:
            config['checkpoint'] += '_%i' % config['pgd_iter']
        if config['eps'] != 8:
            config['checkpoint'] += '_eps=%i' % config['eps']
        if 'pgd' in config['adversary'] or 'fgsm' in config['adversary']:
            # no para alpha if adversary == trades
            if config['alpha'] != 0.5:
                config['checkpoint'] += ('_alpha=%g' % config['alpha']).replace('.', '_')
        if config['adversary'] == 'fat':
            if max(config['fat_taus']) != 2:
                config['checkpoint'] += '_mtau=%i' % max(config['fat_taus'])
        if 'target' in config and config['target']:
            config['checkpoint'] += '_target=%s' % config['target']
        if not config['rand_init']:
            config['checkpoint'] += '_zeroinit'

    if config['lr'] != 0.1:
        config['checkpoint'] += ('_lr=%.e' % config['lr']).replace('.', '_')
    if config['batch_size'] != 128:
        config['checkpoint'] += ('_bs=%i' % config['batch_size']).replace('.', '_')
    if config['wd'] > 0:
        config['checkpoint'] += ('_wd=%g' % config['wd']).replace('.', '_')
    if config['momentum'] > 0:
        config['checkpoint'] += ('_mom=%g' % config['momentum']).replace('.', '_')
    if config['ad_test'] != 'fgsm':
        config['checkpoint'] += '_%s' % config['ad_test']
        if 'pgd' in config['ad_test'] and config['pgd_iter_test'] != 5:
            config['checkpoint'] += '_%i' % config['pgd_iter_test']
    if config['eps_test'] != 8:
        config['checkpoint'] += '_epst=%i' % config['eps_test']
    if config['test']:
        config['checkpoint'] = 'test_' + config['checkpoint']
    del config['test']
    if config['trainsize']:
        config['checkpoint'] += '_ntrain=%i' % config['trainsize']
    if config['testsize']:
        config['checkpoint'] += '_ntest=%i' % config['testsize']
    if 'train_subset_path' in config and config['train_subset_path']:
        config['checkpoint'] += '_sub=%s' % config['train_subset_path'].split('/')[-1].split('.')[0]
    if config['suffix']:
        config['checkpoint'] += '_%s' % config['suffix']
    del config['suffix']

    path = os.path.join('checkpoints', config['checkpoint'])
    path = check_path(path)
    _, checkpoint = os.path.split(path)
    config['checkpoint'] = checkpoint
    # shutil.copy('models.py', path)
    # shutil.copy('config.yaml', path)
    shutil.copytree('src', os.path.join(path, 'src'))

    if config['resume']:
        config['resume_checkpoint'] = 'checkpoint.pth.tar'
        assert(os.path.isfile(os.path.join(path, config['resume_checkpoint']))), 'checkpoint %s not exists!' % config['resume_checkpoint']

    print("\n--------------------------- %s ----------------------------------" % config_file)
    for k, v in config.items():
        print('%s:'%k, v, type(v))
    print("---------------------------------------------------------------------\n")

    return config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch-classification')
    parser.add_argument('--config', '-c', default='config.yaml', type=str, metavar='C', help='config file')
    args = parser.parse_args()

    config = read_config(args.config)
    with open('checkpoints/%s/para.json' % config['checkpoint'], 'w') as f:
        json.dump(config, f)

    # reveal the path to bash
    with open('tmp/path.tmp', 'w') as f:
        f.write(config['checkpoint'])


