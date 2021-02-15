#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.adversary.attacker import dataset_range, clamp, rand_sphere, scale_step

def earlystop(model, data, target, step_size, epsilon, perturb_steps, tau, randominit_type, loss_fn, rand_init=True, omega=0, config=None):
    '''
    The implematation of early-stopped PGD
    Following the Alg.1 in our FAT paper <https://arxiv.org/abs/2002.11242>
    :param step_size: the PGD step size
    :param epsilon: the perturbation bound
    :param perturb_steps: the maximum PGD step
    :param tau: the step controlling how early we should stop interations when wrong adv data is found
    :param randominit_type: To decide the type of random inirialization (random start for searching adv data)
    :param rand_init: To decide whether to initialize adversarial sample with random noise (random start for searching adv data)
    :param omega: random sample parameter for adv data generation (this is for escaping the local minimum.)
    :return: output_adv (friendly adversarial data) output_target (targets), output_natural (the corresponding natrual data), count (average backword propagations count)
    '''

    data_lower = dataset_range[config.dataset]['lower'].to(config.device)
    data_upper = dataset_range[config.dataset]['upper'].to(config.device)

    model.eval()

    K = perturb_steps
    count = 0 # average bp count - not quite useful. record count for each example and see distribution - GAIRAT
    output_target = []
    output_adv = []
    output_natural = []
    # track example, fix the order
    output_indices = []
    output_counts = []

    # control attack steps for each example
    control = (torch.ones(len(target)) * tau).cuda()

    # Initialize the adversarial data with random noise
    if rand_init:
        # if randominit_type == "normal_distribution_randominit":
        if randominit_type == "l2":
            iter_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
            raise NotImplementedError('l2 not supported')
        # if randominit_type == "uniform_randominit":
        if randominit_type == "linf":
            # iter_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda()
            # epsilon is different for differen channel - so have to use our random initialization
            delta = rand_sphere(epsilon, data.size(), device=config.device, norm='linf', requires_grad=False)
            iter_adv = data.detach() + delta
        # iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        iter_adv = clamp(iter_adv, data_lower, data_upper)
    else:
        iter_adv = data.cuda().detach()

    iter_clean_data = data.cuda().detach()
    iter_target = target.cuda().detach()
    output_iter_clean_data = model(data)

    # track example, fix the order
    iter_indices = torch.arange(len(target))

    # record counts of every example
    iter_counts = torch.zeros(len(target))

    while K > 0:
        iter_counts += 1
        # print(iter_counts)
        # print(iter_indices)

        iter_adv.requires_grad_()
        output = model(iter_adv)
        pred = output.max(1, keepdim=True)[1]
        output_index = [] # those examples are finished
        iter_index = [] # those examples keep attacking

        # print(len(pred), len(iter_target))
        # Calculate the indexes of adversarial data those still needs to be iterated
        for idx in range(len(pred)):
            # if pred[idx] != target[idx]:
            if pred[idx] != iter_target[idx]:
                if control[idx] == 0:
                    output_index.append(idx)
                else:
                    control[idx] -= 1
                    iter_index.append(idx)
            else:
                iter_index.append(idx)

        # Add adversarial data those do not need any more iteration into set output_adv
        if len(output_index) != 0:
            if len(output_target) == 0:
                # incorrect adv data should not keep iterated
                output_adv = iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()
                output_natural = iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()
                output_target = iter_target[output_index].reshape(-1).cuda()
                output_indices = iter_indices[output_index]
                output_counts = iter_counts[output_index]
            else:
                # incorrect adv data should not keep iterated
                output_adv = torch.cat((output_adv, iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).cuda()), dim=0)
                output_indices = torch.cat((output_indices, iter_indices[output_index]))
                output_counts = torch.cat((output_counts, iter_counts[output_index]))


        # calculate gradient
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction='mean')(output, iter_target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad


        # update iter adv
        if len(iter_index) != 0:
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            iter_clean_data = iter_clean_data[iter_index]
            iter_target = iter_target[iter_index]
            iter_indices = iter_indices[iter_index]
            iter_counts = iter_counts[iter_index]

            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()

            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().cuda()
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            # iter_adv = torch.clamp(iter_adv, 0, 1)
            iter_adv = clamp(iter_adv, data_lower, data_upper)
            count += len(iter_target)
        else:
            # nothing is left, return
            output_adv = output_adv.detach()
            # return output_adv, output_target, output_natural, count
            # sort out the examples based on the original order
            output_adv = output_adv[output_indices.argsort()]
            assert(torch.all(target == output_target[output_indices.argsort()]))
            output_counts = output_counts[output_indices.argsort()]
            return output_adv, output_counts

        K = K-1

    # concatenate for one more round
    if len(output_target) == 0:
        output_target = iter_target.reshape(-1).squeeze().cuda()
        output_adv = iter_adv.reshape(-1, 3, 32, 32).cuda()
        output_natural = iter_clean_data.reshape(-1, 3, 32, 32).cuda()
        output_indices = iter_indices
        output_counts = iter_counts
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, 32, 32)), dim=0).cuda()
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().cuda()
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, 32, 32).cuda()),dim=0).cuda()
        output_indices = torch.cat((output_indices, iter_indices))
        output_counts = torch.cat((output_counts, iter_counts))

    output_adv = output_adv.detach()
    # sort out the examples based on the original order
    output_adv = output_adv[output_indices.argsort()]
    assert(torch.all(target == output_target[output_indices.argsort()]))
    assert(torch.all(data == output_natural[output_indices.argsort()]))
    output_counts = output_counts[output_indices.argsort()]

    return output_adv, output_counts
