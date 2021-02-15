from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import copy

from ..preprocess import dataset_stats
from ..utils import Dict2Obj, is_in
from ..adversary import attack, scale_step
# from ..models import resnet, resnet_fixup
import src.models as models
from src.models import get_vgg

__all__ = ['DataTool', 'get_ad_examples', 'get_data', 'get_net']

def get_net(path, num_classes, n_channel, device, model='resnet', bn=True, depth=20, width=16, feature=None, state='model.pt', parallel=False):
    state_dict = torch.load('%s/%s' % (path, state), map_location=device)
    if 'vgg' in model:
        net = get_vgg(model=model, batch_norm=bn, num_classes=num_classes, n_channel=n_channel, gain=1.0, dataset='cifar10').to(device)
    elif model in ['ResNet18', 'PreActResNet18', 'FixupPreActResNet18', 'PreActResNetGN18']:
        net = models.__dict__[model]().to(device)
    elif 'resnet' in model:
        net = models.__dict__[model](depth=depth, width=width, num_classes=num_classes, n_channel=n_channel).to(device)
    elif 'wrn' in model:
        net = models.__dict__[model](depth=depth, widen_factor=width, num_classes=num_classes, n_channel=n_channel).to(device)
    else:
        raise KeyError(model)

    if parallel:
        # model is trained on multiple gpu, named after 'module.'
        net = torch.nn.DataParallel(net)

    net.load_state_dict(state_dict)
    net.eval()

    model = net
    if feature:
        feature = copy.deepcopy(net)
        feature.fc = nn.Identity()
        model = {'net': net,
                 'feature': feature,
                 'classifier': copy.deepcopy(net.fc)}
        model = Dict2Obj(model)

    return model


def get_data(loaders, device, feature=None, classes=(('dog',), ('cat',)), n_sample=100):
    assert(len(classes) == 2)
    labels0 = [loaders.class_to_idx[c] for c in classes[0]]
    labels1 = [loaders.class_to_idx[c] for c in classes[1]]

    inputs0, inputs1 = [], []
    for ii, (ins, las) in enumerate(loaders.trainloader):
        print('-- %i' % ii, end='\r')
        ins, las = ins.to(device), las.to(device)
        if feature:
            with torch.no_grad():
                ins = feature(ins)
        inputs0.append(ins[is_in(las, labels0)]) # preds
        inputs1.append(ins[is_in(las, labels1)]) # preds
    inputs0 = torch.cat(inputs0, dim=0)
    inputs1 = torch.cat(inputs1, dim=0)
    inputs0 = inputs0.view(inputs0.size(0), -1)
    inputs1 = inputs1.view(inputs1.size(0), -1)
    print(inputs0.size(), inputs1.size())

    if not n_sample:
        return inputs0, inputs1

    indices0 = np.random.choice(len(inputs0), n_sample, replace=False)
    indices1 = np.random.choice(len(inputs1), n_sample, replace=False)
    return inputs0[indices0], inputs1[indices1]


from ..utils import DeNormalizer
from ..adversary import AAAttacker
def get_ad_examples(net, inputs,
                    labels=None,
                    src_net=None,
                    criterion=nn.CrossEntropyLoss(),
                    eps=40,
                    pgd_alpha=2,
                    pgd_iter=5,
                    dataset='mnist',
                    adversary='fgsm',
                    scale=True,
                    randomize=False,
                    is_clamp=True,
                    target=None,
                    verbose=False,
                    get_prob=False,
                    device=None,
                    path='.'): # For aa log
    """
        a wrapper integrating epsilon scaling
    """

    if src_net is None:
        src_net = net

    # print(type(net))
    # print(type(src_net))

    eps_ = torch.Tensor([eps / 255.]).to(device)
    pgd_alpha_ = torch.Tensor([pgd_alpha / 255.]).to(device)
    if scale:
        eps_ = scale_step(eps, dataset=dataset, device=device)
        pgd_alpha_ = scale_step(pgd_alpha, dataset=dataset, device=device)
    if verbose:
        print('eps: ', eps)
        print('scaled eps: ', eps_.view(-1))

    if labels is None:
        _, labels = net(inputs).max(1)

    if adversary.lower() == 'aa':
        mean = np.array(dataset_stats[dataset]['mean'])
        std = np.array(dataset_stats[dataset]['std'])
        n_channel = inputs.size(1)
        denormalize = DeNormalizer(mean, std, n_channel, device)
        attacker = AAAttacker(net=src_net,
                              eps=eps,
                              normalize=True,
                              mode='fast',
                              path=path, # 
                              # log_path=log_path, # 
                              device=device,
                              data_dir=None) # 
        inputs = denormalize(inputs)
        # attack takes 0-1 inputs
        inputs_ad, _ = attacker.evaluate(x_test=inputs, y_test=labels)
        # attack produces 0-1 outputs
        inputs_ad = attacker._normalize(inputs_ad)
    else:
        inputs_ad = attack(src_net, criterion,
                           inputs, labels,
                           eps=eps_,
                           pgd_alpha=pgd_alpha_,
                           pgd_iter=pgd_iter,
                           adversary=adversary,
                           randomize=randomize,
                           is_clamp=is_clamp,
                           target=target,
                           config=Dict2Obj({'dataset': dataset, 'device': device}))
    logits = net(inputs_ad)
    _, preds_ad = logits.topk(1, 1, True, True)
    preds_ad = preds_ad.squeeze()

    if get_prob:
        probs = F.softmax(logits, dim=1)
        if get_prob == 'pred':
            prob_ad = probs.gather(1, preds_ad.view(-1, 1)).squeeze()
        elif get_prob == 'label':
            prob_ad = probs.gather(1, labels.view(-1, 1)).squeeze()
        else:
            raise KeyError(get_prob)
        if len(prob_ad.shape) == 0:
            return prob_ad.item()
        return inputs_ad, preds_ad, prob_ad.squeeze()

    return inputs_ad, preds_ad


def _geometric_slerp(start, end, interval=(-180, 180), n_mesh=41, split=True):
    assert(len(start.shape) == 1 or start.shape[0] == 1)
    assert(len(end.shape) == 1 or end.shape[0] == 1)
    start = start.ravel()
    end = end.ravel()

    # create an orthogonal basis using QR decomposition
    basis = np.vstack([start, end])
    Q, R = np.linalg.qr(basis.T)
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q.T * signs.T[:, np.newaxis]
    R = R.T * signs.T[:, np.newaxis]

    # calculate the angle between `start` and `end`
    c = np.dot(start, end)
    s = np.linalg.det(R)
    omega = np.arctan2(s, c)

    def to_rad(a):
        return a / 180. * np.pi

    assert(omega > to_rad(interval[0]) and omega < to_rad(interval[1])), ('the interval should include the angle in between!', interval, omega / np.pi * 180)

    # interpolate
    t = np.linspace(to_rad(interval[0]),
                    to_rad(interval[1]),
                    n_mesh)
    start, end = Q
    s = np.sin(t)
    c = np.cos(t)
    interps = start * c[:, np.newaxis] + end * s[:, np.newaxis]

    if split:
        if omega < 0:
            in_between = np.logical_and(t < 0, t > omega)
        else:
            in_between = np.logical_and(t > 0, t < omega)
        # return (interps[in_between], interps[~in_between])

        idx_in = np.argwhere(in_between).ravel()
        idx_s, idx_e = idx_in[0], idx_in[-1]
        return (interps, (idx_s, idx_e))

    return interps


class DataTool:
    def __init__(self, net, length_unit, metric='l2',
                 size=(3, 32, 32), dataset='cifar', device=None):
        self.size = size
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.length_unit = torch.as_tensor(length_unit)
        self.metric = metric

        self.sklearn_metric = {'l2': 'l2', 'linf': 'chebyshev'}.get(metric)
        self.np_metric = {'l2': 2, 'linf': np.inf}.get(metric)
        self.torch_metric = {'l2': 'fro', 'linf': float('inf')}.get(metric)

        self.device = device

    def torch_to_data(self, tensor):
        # assert len(tensor.size()) == 4, tensor.size()
        return tensor.view(tensor.size(0), -1)

    def data_to_torch(self, p):
        if len(p.size()) == 1:
            return p.view(1, *self.size).to(self.device)
        elif len(p.size()) == 2:
            return p.view(p.size(0), *self.size).to(self.device)
        raise KeyError(p.size())

    def to_numpy(self, tensor):
        return tensor.cpu().numpy()

    def to_torch(self, array):
        return torch.from_numpy(array).float().to(self.device)

    def predict_data(self, p, keepdim=False):
        with torch.no_grad():
            _, pred = self.net(self.data_to_torch(p)).topk(1, 1, True, True)
        if keepdim:
            return pred.view(-1)

        if len(p.size()) == 1:
            return pred.item()
        return pred.view(-1)

    def __diff(self, tensor):
        return tensor[1:] - tensor[:-1]

    def decision_check(self, p0, vec, n_mesh=101, scale=True, verbose=True):
        # Sparse check if the decisions along a direction are monotonic
        if scale:
            vec = vec / torch.norm(vec, self.torch_metric) * self.length_unit
        xx = torch.linspace(0, 1, n_mesh).to(self.device)
        ys = self.predict_data(p0 + vec * xx[:, None])
        if torch.all(self.__diff(ys) >= 0) or torch.all(self.__diff(ys) <= 0):
            return True, (xx, ys)
        if verbose:
            print(ys)
        return False, (xx, ys) # Output ys


    def boundary_search(self, p0, vec, n_mesh=101, scale=True, verbose=False,
                        get_mono=False):
        # Binary search
        if scale:
            vec = vec / torch.norm(vec, self.torch_metric) * self.length_unit
            scale = False

        # A coarse exhaustive search
        p0_pred = self.predict_data(p0)
        is_mono, tup = self.decision_check(p0, vec, scale=scale, verbose=verbose)
        xx, ys = tup
        inds_diff = np.argwhere(ys.cpu().numpy() != p0_pred).ravel()
        if inds_diff.size == 0:
            # temporal solution of all consistency case
            if get_mono:
                return xx[-1], True
            return xx[-1]
        ind_first_diff = inds_diff[0]
        xu = xx[ind_first_diff]
        xl = xx[ind_first_diff - 1]

        # Use binary search
        xx = torch.linspace(xl, xu, n_mesh).to(self.device)
        idl = 0
        idu = len(xx) - 1
        idx = idu // 2
        iters = 0
        while idl < idu - 1:
            y = self.predict_data(p0 + xx[idx] * vec)
            if y == p0_pred:
                idl = idx
            else:
                idu = idx
            idx = (idl + idu)  // 2
            if verbose:
                print(iters, idx, idl, idu, y)
            iters += 1

        if get_mono:
            return xx[idx], True
        return xx[idx]


    def get_edge_boundary(self, p0, p1, p2, lim=(0, 1),
                          n_mesh=101, verbose=False, get_mono=False, mode='linear'):

        if mode == 'spherical':
            return self._get_edge_boundary_spherical(p0, p1, p2, lim=lim, n_mesh=n_mesh, verbose=verbose, split=True)

        xx = torch.linspace(*lim, n_mesh).to(self.device)
        interior_points = p1 + (p2 - p1) * xx[:, None]
        bs = []
        monos = []
        for pi in interior_points:
            b, is_mono = self.boundary_search(p0,  pi - p0, get_mono=True)
            bs.append(b)
            monos.append(is_mono)
        if get_mono:
            return bs, monos
        return bs

    def _get_edge_boundary_spherical(self, p0, p1, p2, lim=(-90, 90), n_mesh=101, verbose=False, split=True):

        if split:
            vs, idxs = _geometric_slerp(self.to_numpy(p1 - p0),
            # vs_in, vs_out = _geometric_slerp(self.to_numpy(p1 - p0),
                                        self.to_numpy(p2 - p0),
                                        interval=lim,
                                        n_mesh=n_mesh,
                                        split=split)
            # interps_in, interps_out = p0 + self.to_torch(vs_in), p0 + self.to_torch(vs_out)
            # bs_in = [self.boundary_search(p0,  pi - p0, get_mono=False) for pi in interps_in]
            # bs_out = [self.boundary_search(p0,  pi - p0, get_mono=False) for pi in interps_out]
            # return (bs_in, bs_out)

            interps = p0 + self.to_torch(vs) 
            bs = [self.boundary_search(p0,  pi - p0, get_mono=False) for pi in interps]
            return (bs, idxs)

        interps = p0 + self.to_torch(_geometric_slerp(self.to_numpy(p1 - p0),
                                                      self.to_numpy(p2 - p0),
                                                      interval=lim,
                                                      n_mesh=n_mesh,
                                                      split=split))
        assert len(interps) == n_mesh, (len(interps), n_mesh)
        return [self.boundary_search(p0,  pi - p0, get_mono=False) for pi in interps]

    def get_boundary_points(self, p0, points):
        brs_ooc = [self.boundary_search(p0, p - p0, scale=False) for p in points]
        bps_ooc = [p0 + (p - p0) * r for r, p in zip(brs_ooc, points)]
        return bps_ooc    
    
    def get_edge_boundary_points(self, p0, p1, p2, lim=(0, 1), n_mesh=101, verbose=False):
        xx = torch.linspace(*lim, n_mesh).to(self.device)
        interior_points = p1 + (p2 - p1) * xx[:, None]
        return self.get_boundary_points(p0, interior_points)


    def attack_data(self, p, label=None, eps=2, adversary='fgsm', pgd_alpha=2, pgd_iter=5, randomize=False, pred=True, verbose=False, scale=True, is_clamp=True, path='.'):
        p_ad, pred_ad = get_ad_examples(self.net,
                                        self.data_to_torch(p),
                                        label,
                                        criterion=self.criterion,
                                        eps=eps,
                                        pgd_alpha=pgd_alpha,
                                        pgd_iter=pgd_iter,
                                        adversary=adversary,
                                        dataset=self.dataset,
                                        scale=scale,
                                        randomize=randomize,
                                        is_clamp=is_clamp,
                                        verbose=verbose,
                                        device=self.device,
                                        path=path)
        if pred:
            return self.torch_to_data(p_ad), pred_ad
        return self.torch_to_data(p_ad)


    def predict_logits(self, p):
        logit, pred = self.net(self.data_to_torch(p)).topk(1, 1, True, True)
        # print(logit, pred)
        # assert(logit > 0), logit
        mask = pred * 2 - 1 # 0, 1 -> -1, 1
        logit *= mask
        if len(p.shape) == 1:
            return logit.item()
        return logit.cpu().detach().numpy().ravel()

    def predict_probs(self, p, binarize=False):
        logits = self.net(self.data_to_torch(p))
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.squeeze()
        probs = F.softmax(logits, dim=1)
        prob = probs.gather(1, pred.view(-1, 1)).squeeze()
        if binarize:
            # project to 10 bins for plotting
            prob = pred + prob
        if len(prob.shape) == 0:
            return prob.item()
        return prob.cpu().detach().numpy().ravel()


    # analyze
    def distance(self, p1, p2=None, verbose=True):
        p1 = p1.cpu().numpy()
        if len(p1.shape) == 1:
            p1 = p1[None, :]
        if p2 is not None:
            p2 = p2.cpu().numpy()
            if len(p2.shape) == 1:
                p2 = p2[None, :]
        return np.squeeze(pairwise_distances(p1, p2) / self.length_unit.item())

    def pairwise_angle(self, p1, p2=None):
        p1 = p1.cpu().numpy()
        if len(p1.shape) == 1:
            p1 = p1[None, :]
        if p2 is not None:
            p2 = p2.cpu().numpy()
            if len(p2.shape) == 1:
                p2 = p2[None, :]
        cossim = cosine_similarity(p1, p2) # .ravel()
        cossim[np.abs(cossim) > 1] = 1
        return np.squeeze(np.arccos(cossim) / np.pi * 180)
