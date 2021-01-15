import argparse
import numpy as np
import yaml
import torch
import re
import sys
from utils import *
from dataloader.dataloader import get_dataloader
from dataloader.transform import parse_policies, MultiAugmentation
from models import *
from optimizer_scheduler import get_optimizer_scheduler
from tqdm import tqdm

seed_everything(0)
def parse_args():
    parser = argparse.ArgumentParser(description='Unofficial Implementation of Adversarial Autoaugment')
    parser.add_argument('--load_conf', type = str)
    parser.add_argument('--logdir', type = str)
    parser.add_argument('--M', type = int)
    parser.add_argument('--local_rank', type = int, default = -1)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    return args

def init_ddp(local_rank):
    if local_rank !=-1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl',init_method='env://')

def load_yaml(dir):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    conf = yaml.load(open(dir, 'r'), Loader=loader)
    return conf
        
if __name__ == '__main__':
    args = parse_args()
    
    conf = load_yaml(args.load_conf)
    logger = Logger(os.path.join(args.logdir,args.load_conf.split('/')[-1].split('.')[0]))
    num_gpus = torch.cuda.device_count()    
    
    ## DDP set print_option + initialization
    if args.local_rank > 0:
        sys.stdout = open(os.devnull, 'w')    
    init_ddp(args.local_rank)
    print("EXPERIMENT:",args.load_conf.split('/')[-1].split('.')[0])
    print()
    
    train_sampler, train_loader, valid_loader, test_loader = get_dataloader(conf, dataroot = './dataloader/datasets', split = 0.15, split_idx = 0, multinode = (args.local_rank!=-1))
    
    controller = get_controller(conf,args.local_rank)
    model = get_model(conf,args.local_rank)
    (optimizer, scheduler), controller_optimizer = get_optimizer_scheduler(controller, model, conf)
    criterion = CrossEntropyLabelSmooth(num_classes = num_class(conf['dataset']))
    
    if args.amp:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
    
    step = 0
    for epoch in range(conf['epoch']):
        if args.local_rank >=0:
            train_sampler.set_epoch(epoch)
        
        Lm = torch.zeros(args.M).cuda()
        Lm.requires_grad = False
        
        policies, log_probs, entropies = controller(args.M) # (M,2*2*5) (M,) (M,) 
        policies = policies.cpu().detach().numpy()
        parsed_policies = parse_policies(policies)        
        trfs_list = train_loader.dataset.dataset.transform.transforms 
        trfs_list[2] = MultiAugmentation(parsed_policies)## replace augmentation into new one
        
        controller_optimizer.zero_grad()
        
        train_loss = 0
        train_top1 = 0
        train_top5 = 0
        
        progress_bar = tqdm(train_loader)
        for idx, (data,label) in enumerate(progress_bar):
            optimizer.zero_grad()
            data = data.cuda()
            label = label.cuda()
            with autocast():
                pred = model(data)
                losses = [criterion(pred[i::args.M,...] ,label) for i in range(args.M)]
                loss = torch.mean(torch.stack(losses))
                
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            for i,_loss in enumerate(losses):
                _loss = _loss.detach()
                Lm[i] += _loss / len(train_loader)
            
            top1, top5 = accuracy(pred, torch.cat([label]*args.M,dim=0), (1, 5))
            train_loss += reduced_metric(loss.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            train_top1 += reduced_metric(top1.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            train_top5 += reduced_metric(top5.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            
            progress_bar.set_description('Step: {}. LR : {:.5f}. Epoch: {}/{}. Iteration: {}/{}. Train_Loss : {:.5f}'.format(step, optimizer.param_groups[0]['lr'], epoch, conf['epoch'], idx + 1, len(train_loader), loss.item()))
            step += 1
            
        normalized_Lm = (Lm - torch.mean(Lm))/(torch.std(Lm) + 1e-6)
        controller_loss = -log_probs * normalized_Lm - conf['entropy_penalty'] * entropies
        controller_loss = torch.mean(controller_loss)
        controller_loss.backward()
        controller_optimizer.step()
        scheduler.step()
        
        valid_loss = 0.
        valid_top1 = 0.
        valid_top5 = 0.
        with torch.no_grad():
            for idx, (data,label) in enumerate(tqdm(valid_loader)):
                data = data.cuda()
                label = label.cuda()
                
                pred = model(data)
                loss = criterion(pred,label)

                top1, top5 = accuracy(pred, label, (1, 5))                
                valid_loss += reduced_metric(loss.detach(), num_gpus, args.local_rank !=-1) / len(valid_loader)
                valid_top1 += reduced_metric(top1.detach(), num_gpus, args.local_rank !=-1) / len(valid_loader)
                valid_top5 += reduced_metric(top5.detach(), num_gpus, args.local_rank !=-1) / len(valid_loader)
        
        logger.add_dict(
            {
                'train_loss' : train_loss,
                'train_top1' : train_top1,
                'train_top5' : train_top5,
                'controller_loss' : controller_loss.item(),
                'valid_loss' : valid_loss,
                'valid_top1' : valid_top1,
                'valid_top5' : valid_top5,
                'policies' : parsed_policies,
            }
        )
        if args.local_rank <= 0:
            logger.save_model(model,epoch)
        logger.info(epoch,['train_loss','train_top1','train_top5','valid_loss','valid_top1','valid_top5','controller_loss'])
    
    logger.save_logs()