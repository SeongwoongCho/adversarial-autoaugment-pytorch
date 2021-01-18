import os

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist

from torch.utils.data import Subset, ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit
from .imagenet import ImageNet
from .transform import get_basetransform, train_collate_fn, test_collate_fn

def get_dataloader(conf, dataroot = './dataloader/datasets', split = 0.15, split_idx = 0, multinode = False):
    batch_size = conf['batch_size']
    transform_train, transform_test, transform_target = get_basetransform(conf['dataset'])
    
    if conf['dataset'] == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train, target_transform = transform_target)
        validset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_test, target_transform = transform_target)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test, target_transform = transform_target)
    elif conf['dataset'] == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train, target_transform = transform_target)
        validset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_test, target_transform = transform_target)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test, target_transform = transform_target)
    elif conf['dataset'] == 'svhn':
        trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_train, target_transform = transform_target)
        # train_extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=True, transform=transform_train)
        # trainset = ConcatDataset([trainset, train_extraset])
        validset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_test, target_transform = transform_target)
        # valid_extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=True, transform=transform_test)
        # validset = ConcatDataset([validset, valid_extraset])
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test, target_transform = transform_target)
    elif conf['dataset'] == 'imagenet':
        trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'),download = True, transform=transform_train, target_transform = transform_target)
        validset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'),download = True, transform=transform_test, target_transform = transform_target)
        testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'),download = True, split='val', transform=transform_test, target_transform = transform_target)
        
        # compatibility
        trainset.targets = [lb for _, lb in trainset.samples]
        validset.targets = [lb for _, lb in validset.samples]
    else:
        raise Exception()
    
    if split > 0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(trainset))), trainset.labels if conf['dataset'] == 'svhn' else trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        trainset = Subset(trainset,train_idx)
        validset = Subset(validset,valid_idx)
    else:
        trainset = Subset(trainset, list(range(len(trainset.labels if conf['dataset'] == 'svhn' else trainset.targets))))
        validset = Subset(validset,[])
    train_sampler = None
    valid_sampler = None
    test_sampler = None
    if multinode:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
                validset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        test_sampler = torch.utils.data.distributed.DistributedSampler(
                testset, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True if train_sampler is None else False, num_workers=8, pin_memory=True,
        sampler=train_sampler, drop_last=True, collate_fn = train_collate_fn)
    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
        sampler=valid_sampler, drop_last=False, collate_fn = test_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
        sampler=test_sampler, drop_last=False, collate_fn = test_collate_fn
    )
    
    return train_sampler, train_loader, valid_loader, test_loader