import torch

from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np

from .resnet import ResNet
from .pyramidnet import PyramidNet
from .shakeshake.shake_resnet import ShakeResNet
from .wideresnet import WideResNet
from .shakeshake.shake_resnext import ShakeResNeXt
from .controller import Controller
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

def get_controller(conf, local_rank=-1):
    model = Controller()
    if local_rank >= 0:
        device = torch.device('cuda', local_rank)
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = model.cuda()
        model = DataParallel(model)
        model.sample = model.module.sample
    cudnn.benchmark = True
    return model

def get_model(conf, local_rank=-1):
    name = conf['model']
    num_classes = num_class(conf['dataset'])
    
    if name == 'resnet50':
        model = ResNet(dataset='imagenet', depth=50, num_classes=num_classes, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', depth=200, num_classes=num_classes, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_classes)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_classes)

    elif name == 'shakeshake26_2x32d':
        model = ShakeResNet(26, 32, num_classes)
    elif name == 'shakeshake26_2x64d':
        model = ShakeResNet(26, 64, num_classes)
    elif name == 'shakeshake26_2x96d':
        model = ShakeResNet(26, 96, num_classes)
    elif name == 'shakeshake26_2x112d':
        model = ShakeResNet(26, 112, num_classes)
    elif name == 'shakeshake26_2x96d_next':
        model = ShakeResNeXt(26, 96, 4, num_classes)

    elif name == 'pyramid':
        model = PyramidNet(conf['dataset'], depth=conf['depth'], alpha=conf['alpha'], num_classes=num_classes, bottleneck=conf['bottleneck'])

    if local_rank >= 0:
        device = torch.device('cuda', local_rank)
        model = model.cuda()
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters = True)
    else:
        model = model.cuda()
        model = DataParallel(model)
    cudnn.benchmark = True
    
    return model

def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]