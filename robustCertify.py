#!./env python

# from autoattack import AutoAttack
import torch
import numpy as np
import os
import argparse

import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

from src.adversary import AAAttacker
from src.utils import get_net, str2bool

def robust_certify(model, depth, width, model_parallel=False, normalize=True,
                   path='.', state='last', gpu_id='0', # sample=1000, seed=7,
                   mode='standard',
                   data_dir='/home/chengyu/RobustDataProfiling/data'):

    ## Current setting: evaluate on a random subset of 1000 (fixed during training)
    ## Fast setting for epoch-wise evaluation: same as above but use agpd-t only
    ## Leaderboard evalulation setting: n_ex=10000, i.e. use the entire testset

    # standard: all four attack, entire test set
    # fast: first two attack, entire test set
    assert(mode in ['standard', 'fast'])

    print('>>>>>>>>>>> set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('>>>>>>>>>>> get net..')
    if state == 'last':
        model_state = 'model.pt'
    elif state == 'best':
        model_state = 'best_model.pt'
    else:
        raise KeyError(state)

    log_path = 'log_certify_%s' % state
    if mode != 'standard':
        log_path += '_%s' % mode
    log_path += '.txt'

    net = get_net(path,
                  num_classes=10,
                  n_channel=3,
                  feature=None,
                  model=model,
                  depth=depth,
                  width=width,
                  state=model_state,
                  parallel=model_parallel,
                  device=device)

    print('>>>>>>>>>>> start evaluating..')
    attacker = AAAttacker(net=net,
                          normalize=normalize,
                          mode=mode,
                          path=path,
                          log_path=log_path,
                          device=device,
                          data_dir=data_dir)
    attacker.evaluate()

    print('>>>>>>>>>>> Done.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", default='resnet', type=str, help='model')
    parser.add_argument('--depth', default=20, type=int, help='model depth')
    parser.add_argument('--width', default=64, type=int, help='model width')
    parser.add_argument("-mp", "--model-parallel", type=str2bool, nargs='?', const=True, default=False, help="model parallel?")
    parser.add_argument("--norm", type=str2bool, nargs='?', const=True, default=False, help="normalized inputs?")
    parser.add_argument("-p", "--path", type=str, help="model path")
    parser.add_argument('-d', "--state", default='last', type=str, help='model state')
    parser.add_argument("-g", "--gpu", default='0', type=str, help="gpu_id")
    parser.add_argument("--mode", default='standard', type=str, help="eval mode")
    args = parser.parse_args()

    print(args.model_parallel)

    robust_certify(model=args.model, depth=args.depth, width=args.width, model_parallel=args.model_parallel, normalize=args.norm,
                   path=args.path, state=args.state, gpu_id=args.gpu, mode=args.mode)
