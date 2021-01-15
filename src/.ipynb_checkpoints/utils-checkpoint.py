import random
import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import torch.distributed as dist
import copy
from collections import defaultdict

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True    

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    return res
    
class CrossEntropyLabelSmooth(torch.nn.Module):
    def __init__(self, num_classes, epsilon = 0, reduction='mean'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        log_probs = self.logsoftmax(input)
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        if self.epsilon > 0.0:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        targets = targets.detach()
        loss = (-targets * log_probs)
                
        if self.reduction in ['avg', 'mean']:
            loss = torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class Logger:
    def __init__(self,log = './logs'):
        self.metrics = defaultdict(lambda:[])
        self.log = log
        
        os.makedirs(self.log,exist_ok=True)
        os.makedirs(os.path.join(self.log,'models/'),exist_ok=True)
        
    def add(self, key, value):
        self.metrics[key].append(value)

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))
    
    def save_model(self,model, epoch):
        save_path = os.path.join(self.log,'models/')
        torch.save(model.module.state_dict(), os.path.join(save_path,'checkpoint_%d.pth'%epoch))
        if self.metrics['valid_loss'][-1] == np.min(self.metrics['valid_loss']):
            torch.save(model.module.state_dict(), os.path.join(save_path,'best.pth'))
    
    def save_logs(self):
        save_path = os.path.join(self.log,'logs.pkl')        
        open_file = open(save_path, "wb")
        pickle.dump(self.get_dict(), open_file)
        open_file.close()
    
    def info(self, epoch, keys):
        output = "Epoch: {:d} ".format(epoch)
        for key in keys:
            output += "{}: {:.5f} ".format(key, self.metrics[key][-1])
        print(output)
        
def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt

def reduced_metric(metric,num_gpus,ddp=True):
    if ddp:
        reduced_loss = reduce_tensor(metric.data, num_gpus)
        return reduced_loss.item()
    return loss.item()