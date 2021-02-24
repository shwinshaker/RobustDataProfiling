#!./env python

import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['get_vgg', 'vgg11', 'vgg11_bn', 'vgg11_lite', 'vgg19', 'vgg19_bn']

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, gain=1.0, out=512):
        super(VGG, self).__init__()
        self.gain = gain
        self.features = features
        self.classifier = nn.Linear(out, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                # assert(n == fan_in)
                assert(n == fan_out)
                m.weight.data.normal_(0, self.gain * math.sqrt(2. / n))
                # m.weight.data.normal_(0, self.gain * math.sqrt(1. / n)) # sometimes try this, will converge..
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, n_channel=3, batch_norm=False):
    layers = []
    in_channels = n_channel
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            # TODO: bias should be false, but conflicted with parameter tracker
            layers += [nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A1': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    # 'A': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M'],
    # 'A_mnist': {'arch': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M'], 'out': 64},
    # 'A_mnist': {'arch': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M'], 'out': 128},
    # 'A_mnist': {'arch': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M'], 'out': 128},
    # 'A_mnist': {'arch': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'], 'out': 512},
    'A_mnist': {'arch': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'], 'out': 512},
    # 'A_mnist': {'arch': [64, 64, 64, 64, 64, 64], 'out': },

    'A': {'arch': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'out': 512},
     # 'A': {'arch': [64, 'A', 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'A'], 'out': 512},
    'B': {'arch': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'out': 512},
    'D': {'arch': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'out': 512},
    'E': {'arch': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], 'out': 512},
}

def get_vgg(model, **kwargs):
    if model == 'vgg11':
        if kwargs['batch_norm']:
            return vgg11_bn(**kwargs)
        else:
            return vgg11(**kwargs)

    # if model == 'vgg11_lite':
    #     return vgg11_lite(**kwargs)

    if model == 'vgg19':
        if kwargs['batch_norm']:
            return vgg19_bn(**kwargs)
        else:
            return vgg19(**kwargs)

    raise KeyError(model)


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    c = 'A'
    if 'cifar' not in kwargs['dataset']:
        c += '_%s' % kwargs['dataset']

    model = VGG(make_layers(cfg[c]['arch'], n_channel=kwargs['n_channel'], batch_norm=False),
                out=cfg[c]['out'], num_classes=kwargs['num_classes'], gain=kwargs['gain'])
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    # model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    c = 'A'
    model = VGG(make_layers(cfg[c]['arch'], n_channel=kwargs['n_channel'], batch_norm=True),
                out=cfg[c]['out'], num_classes=kwargs['num_classes'], gain=kwargs['gain'])
    return model

def vgg11_lite(**kwargs):
    model = VGG(make_layers(cfg['A1']), out=64, **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = VGG(make_layers(cfg['E']), **kwargs)
    c = 'E'
    model = VGG(make_layers(cfg[c]['arch'], n_channel=kwargs['n_channel'], batch_norm=False),
                out=cfg[c]['out'], num_classes=kwargs['num_classes'], gain=kwargs['gain'])
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    # model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    c = 'E'
    model = VGG(make_layers(cfg[c]['arch'], n_channel=kwargs['n_channel'], batch_norm=True),
                out=cfg[c]['out'], num_classes=kwargs['num_classes'], gain=kwargs['gain'])
    return model
