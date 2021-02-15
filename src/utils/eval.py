#!./env python

import torch
import torch.nn as nn

__all__ = ['test', 'AverageMeter', 'GroupMeter', 'accuracy', 'alignment', 'criterion_r']


def test(testloader, net, criterion, config, classes=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    net.eval()

    if hasattr(config, 'class_eval') and config.class_eval:
        top1_class = GroupMeter(classes)
    
    for i, tup in enumerate(testloader, 0):
        if len(tup) == 2:
            inputs, labels = tup
        else:
            inputs, labels, _ = tup
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        prec1, = accuracy(outputs.data, labels.data)

        losses.update(loss.item(), inputs.size(0))        
        top1.update(prec1.item(), inputs.size(0))

        if hasattr(config, 'class_eval') and config.class_eval:
            top1_class.update(outputs, labels)

    extra_metrics = dict()
    if hasattr(config, 'class_eval') and config.class_eval:
        extra_metrics['class_acc'] = top1_class.output_group()
        
    return losses.avg, top1.avg, extra_metrics


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GroupMeter:
    """
        measure the accuracy of each class
    """
    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
        self.meters = [AverageMeter() for _ in range(self.num_classes)]

    def update(self, out, las):
        _, preds = out.topk(1, 1, True, True)
        preds = preds.squeeze()
        for c in range(self.num_classes):
            num_c = (las == c).sum().item()
            if num_c == 0:
                continue
            acc = ((preds == las) & (las == c)).sum() * 100. / num_c
            self.meters[c].update(acc.item(), num_c)

    def output(self):
        return np.mean(self.output_group())

    def output_group(self):
        return [self.meters[i].avg for i in range(self.num_classes)]

    def pprint(self):
        print('')
        for i in range(self.num_classes):
            print('%10s: %.4f' % (self.classes[i], self.meters[i].avg))


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def alignment(output1, output2):
    batch_size = output1.size(0)

    _, pred1 = output1.topk(1, dim=1, largest=True)
    pred1 = pred1.t()
    _, pred2 = output2.topk(1, dim=1, largest=True)
    pred2 = pred2.t()
    correct = pred1.eq(pred2)
    return correct.view(-1).float().sum(0).mul_(100.0 / batch_size)


def criterion_r(output1, output2, c=None):

    if isinstance(c, nn.CrossEntropyLoss):
        # https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720/17
        def cross_entropy(pred, soft_targets):
            logsoftmax = nn.functional.log_softmax
            return torch.mean(torch.sum(- soft_targets * logsoftmax(pred, dim=1), 1))
        return cross_entropy(output1, output2)
        
    raise NotImplementedError(type(c))

