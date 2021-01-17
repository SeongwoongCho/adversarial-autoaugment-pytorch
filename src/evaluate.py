import torch
import argparse

from dataloader.dataloader import get_dataloader
from models import *
from utils import *
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--load_conf', type = str)
    parser.add_argument('--logdir', type = str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    conf = load_yaml(args.load_conf)
    
    train_sampler, train_loader, valid_loader, test_loader = get_dataloader(conf, dataroot = './dataloader/datasets', split = 0.01, split_idx = 0, multinode = False)
    
    ckpt = os.path.join(os.path.join(args.logdir,args.load_conf.split('/')[-1].split('.')[0]),'models/checkpoint_177.pth')
    state_dict = torch.load(ckpt)
    model = get_model(conf,-1, state_dict)
    model.eval()
    criterion = CrossEntropyLabelSmooth(num_classes = num_class(conf['dataset']))
    
    test_loss = 0.
    test_top1 = 0.
    test_top5 = 0.
    
    cnt = 0.
    with torch.no_grad():
        for idx, (data,label) in enumerate(tqdm(test_loader)):
            b = data.size(0)
            data = data.cuda()
            label = label.cuda()

            pred = model(data)
            loss = criterion(pred,label)
            
            top1, top5 = accuracy(pred, label, (1, 5))
            
            test_loss += loss.item()*b
            test_top1 += top1.item()*b
            test_top5 += top5.item()*b
            
            cnt += b
            
        test_loss = test_loss / cnt
        test_top1 = test_top1 / cnt
        test_top5 = test_top5 / cnt
    
    print("TEST LOSS:",test_loss)
    print("TEST TOP1:",test_top1)
    print("TEST TOP5:",test_top5)