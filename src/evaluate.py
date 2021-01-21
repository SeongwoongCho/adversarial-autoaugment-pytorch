import torch
import argparse
import wget 

from dataloader.dataloader import get_dataloader
from models import *
from utils import *
from tqdm import tqdm

archive = {
    'cifar10-C' : 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1',
    'cifar10-P' : 'https://zenodo.org/record/2535967/files/CIFAR-10-P.tar?download=1',
    'cifar100-C' : 'https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1',
}

def download_corruption_dataset(dataset, root = './dataloader/datasets/'):
    if 'cifar10' in dataset:
        if not os.path.exists(os.path.join(root,'CIFAR-10-C')):
            wget.download(archive['cifar10-C'],out=root)
            extract_tar(os.path.join(root,'CIFAR-10-C.tar'))
        if not os.path.exists(os.path.join(root,'CIFAR-10-P')):
            wget.download(archive['cifar10-P'],out=root)
            extract_tar(os.path.join(root,'CIFAR-10-P.tar'))
    if 'cifar100' in dataset:
        if not os.path.exists(os.path.join(root,'CIFAR-100-C')):
            wget.download(archive['cifar100-C'],out=root)
            extract_tar(os.path.join(root,'CIFAR-100-C.tar'))
    if 'imagenet' in dataset:
        pass
    
def bar_custom(current, total, width=80):
    width=30
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d]" % (current / total * 100, percent_bar, current, total)
    return progress
    
def extract_tar(src, dest=None, gzip=None, delete=False):
    import tarfile

    if dest is None:
        dest = os.path.dirname(src)
    if gzip is None:
        gzip = src.lower().endswith('.gz')

    mode = 'r:gz' if gzip else 'r'
    with tarfile.open(src, mode) as tarfh:
        tarfh.extractall(path=dest)

    if delete:
        os.remove(src)
    
def parse_args():
    parser = argparse.ArgumentParser(description='EVALUATION')
    parser.add_argument('--load_conf', type = str)
    parser.add_argument('--logdir', type = str)
    parser.add_argument('--seed', type = int, default = 0)
    args = parser.parse_args()
    return args

def test(model, test_loader, criterion):
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
    
    return test_loss,test_top1, test_top5
    
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

if __name__ == '__main__':
    args = parse_args()
    conf = load_yaml(args.load_conf)
    
    train_sampler, train_loader, valid_loader, test_loader = get_dataloader(conf, dataroot = './dataloader/datasets', split = 0.01, split_idx = 0, multinode = False)
    
#    ckpt = os.path.join(os.path.join(args.logdir,args.load_conf.split('/')[-1].split('.')[0]),'models/best_loss.pth')
    ckpt = os.path.join(os.path.join(args.logdir,args.load_conf.split('/')[-1].split('.')[0]) + "_%d"%args.seed,'models/best_top1.pth')
    state_dict = torch.load(ckpt)
    model = get_model(conf,-1, state_dict)
    model.eval()
    criterion = CrossEntropyLabelSmooth(num_classes = num_class(conf['dataset']))
    
    print("CIFAR10 TESTSET RESULT")
    test(model,test_loader,criterion)
    print()
    
    
    """
    Test Corruption Dataset
    """
        
    download_corruption_dataset(conf['dataset'], root = './dataloader/datasets/')
    
    if conf['dataset'] == 'cifar10':
        print("CIFAR10-C TESTSET RESULT")
        corruption_losses = []
        corruption_top1s = []
        corruption_top5s = []
        base_path = './dataloader/datasets/CIFAR-10-C/'
        for corruption in CORRUPTIONS:
            # Reference to original data is mutated
            test_loader.dataset.data = np.load(base_path + corruption + '.npy')
            test_loader.dataset.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
            print("====CORRUPTION====")
            print(corruption)
            print("==================")
            loss, top1,top5 = test(model,test_loader,criterion)
            corruption_losses.append(loss)
            corruption_top1s.append(top1)
            corruption_top5s.append(top5)
        
        print()
        print("TEST MEAN CORRUPTION LOSS:",np.mean(corruption_losses))
        print("TEST MEAN CORRUPTION TOP1:",np.mean(corruption_top1s))
        print("TEST MEAN CORRUPTION TOP5:",np.mean(corruption_top5s))