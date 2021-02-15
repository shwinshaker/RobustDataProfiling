#!./env python

import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt
import os
from collections.abc import Iterable
import matplotlib.patches as mpatches
import warnings

__all__ = ['load_log', 'plot']

def load_log(logfile, nlogs=None, window=1, interval=None):
    if not os.path.isfile(logfile):
        warnings.warn('%s not found.' % logfile)
        return None

    with open(logfile, 'r') as f:
        header = f.readline().strip().split()
    data = np.loadtxt(logfile, skiprows=1)
    if data.size == 0:
        return None
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)

    if interval is not None:
        data = data[::interval]
    if not nlogs: nlogs = data.shape[0]
    data = data[:nlogs]

    def smooth(record):
        return bn.move_mean(record, window=window)
    return dict([(h, smooth(data[:, i])) for i, h in enumerate(header)])

def plot(dic, colors=None, interval=None,
         window=1, ylim=None, options=['clean', 'ad'],
         phases=['train', 'test'], metric='Acc',
         # lengend_loc='best',
         train_mode='test',
         offsets=[],
         logx=False,
         logy=False):


    assert(metric in ['Acc', 'Err']), metric
    def trans(arr):
        if metric == 'Acc':
            return arr
        if metric == 'Err':
            return 100 - arr

    ## -- train modes --
    # test: test on a subset from the entire training set
    # extra_test: test on a subset from the current training set
    # extra: original training accuracy
    if isinstance(train_mode, str):
        assert(train_mode in ['test', 'extra_test', 'extra'])
    elif isinstance(train_mode, list):
        assert(all([m in ['test', 'extra_test', 'extra'] for m in train_mode]))

    if colors is None:
        colors = ['C%i' % i for i in range(len(dic))]
    fig, ax = plt.subplots()
    legend_dict = dict()
    for i, (label, path) in enumerate(dic.items()):
        window_ = window
        if isinstance(window, Iterable):
            window_ = window[i]

        interval_ = interval
        if isinstance(interval, Iterable):
            interval_ = interval[i]

        stats = load_log(os.path.join(path, 'log.txt'), window=window_, interval=interval_)
        if stats is None:
            continue

        if isinstance(train_mode, str):
            train_mode_ = train_mode
        elif isinstance(train_mode, list):
            train_mode_ = train_mode[i]

        if train_mode_ == 'test':
            stat_train = trans(stats['Train-Acc'])
            stat_train_ad = trans(stats['Train-Acc-Ad'])
        elif train_mode_ == 'extra':
            stats_ = load_log(os.path.join(path, 'log_extra.txt'), window=window_, interval=interval_)
            stat_train = trans(stats_['Train-Acc'])
            stat_train_ad = trans(stats_['Train-Acc-Ad'])
        elif train_mode_ == 'extra_test':
            stats_ = load_log(os.path.join(path, 'log_trainEval.txt'), window=window_, interval=interval_)
            stat_train = trans(stats_['Train-Acc-0'])
            stat_train_ad = trans(stats_['Train-Acc-Ad-0'])
        else:
            raise KeyError(train_mode_)

        plotx = np.arange(len(stats['Train-Acc']))
        if offsets:
            plotx += offsets[i]

        if 'clean' in options:
            if 'train' in phases:
                ax.plot(plotx, stat_train, color=colors[i], linestyle='-.')
            if 'test' in phases:
                ax.plot(plotx, trans(stats['Test-Acc']), color=colors[i], linestyle=':')
        if 'ad' in options:
            if 'train' in phases:
                ax.plot(plotx, stat_train_ad, color=colors[i], linestyle='-')
            if 'test' in phases:
                ax.plot(plotx, trans(stats['Test-Acc-Ad']), color=colors[i], linestyle='--')
        legend_dict[label] = colors[i]


    if ylim:
        ax.set_ylim(ylim)

    # ax.legend(loc=lengend_loc, bbox_to_anchor=(1, 1))
    # Manually set the legend
    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)
    ax.legend(handles=patchList, bbox_to_anchor=(1, 1))

    fontsize = 15
    ax.set_xlabel('Epochs', fontsize=fontsize)
    if metric == 'Acc':
        ax.set_ylabel('Accuracy (%)', fontsize=fontsize)
    else:
        ax.set_ylabel('Error (%)', fontsize=fontsize)

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
