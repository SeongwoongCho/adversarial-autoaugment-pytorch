import torch
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR

def get_optimizer_scheduler(controller, model, conf):
    ## define controller optimizer
    controller_optimizer = Adam(controller.parameters(), lr = 0.00035)
    
    ## define optimizer
    optimizer = SGD(model.parameters(), lr = conf['lr'], momentum = 0.9, nesterov = True, weight_decay = conf['weight_decay'])
    
    ## define scheduler
    if conf['model'] == 'wresnet28_10':
        scheduler = MultiStepLR(optimizer, [60,120,160], gamma=0.2, last_epoch=-1, verbose=False)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max = conf['epoch'])
        
    return (optimizer, scheduler), controller_optimizer
