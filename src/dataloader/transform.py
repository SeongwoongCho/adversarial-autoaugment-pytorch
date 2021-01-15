from torchvision.transforms import transforms
from .augmentations import *

NUM_MAGS = 10
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
_IMAGENET_MEAN, _IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def get_basetransform(dataset):
    if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'svhn':
        normalize = transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
        cutout = 16 if 'cifar' in dataset else 20
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x), ## Locate new policy
            transforms.Lambda(lambda imgs: torch.stack([CutoutDefault(cutout)(normalize(transforms.ToTensor()(img))) for img in imgs]))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        transform_target = lambda target : target
        
    elif dataset == 'imagenet':
        image_size = 224
        normalize = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size,interpolation=Image.BICUBIC), 
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x), ## Locate new policy
            transforms.Lambda(lambda imgs: torch.stack([normalize(transforms.ToTensor()(img)) for img in imgs]))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(image_size + 32, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])
        transform_target = lambda target : target
        
    return transform_train, transform_test, transform_target

class Augmentation(object):
    def __init__(self, policy):
        """
        For example, policy is [[(op, mag), (op,mag)]]*Q
        """
        self.policy = policy

    def __call__(self, img):
        sub_policy = random.choice(self.policy)
        for op,mag in sub_policy:
            img = apply_augment(img, op, mag)
        return img
    
class MultiAugmentation(object):
    def __init__(self, policies):
        self.policies = [Augmentation(policy) for policy in policies]

    def __call__(self,img):
        imgs = [policy(img) for policy in self.policies] 
        return imgs

def parse_policies(policies):
    # policies : (M,4(op,mag,op,mag)*5(sub_policy))
    # parsed_policies : [[[(op, mag), (op,mag)]]*5] * M
    
    al = augment_list()
    
    M, S = policies.shape
    S = S//4
    parsed_policies = []
    for i in range(M):
        parsed_policy = []
        for j in range(S):
            parsed_policy.append([(al[(policies[i][4*j])][0].__name__,policies[i][4*j+1]/NUM_MAGS),(al[policies[i][4*j+2]][0].__name__,policies[i][4*j+3]/(NUM_MAGS-1))])
        parsed_policies.append(parsed_policy)
    
    return parsed_policies

def train_collate_fn(batch):
    """
    batch = [((M,3,H,W), label)]*batch_size
    """
    
    data = torch.cat([b[0] for b in batch], dim = 0)
    label = torch.tensor([b[1] for b in batch])
    
    return data,label

def test_collate_fn(batch):
    """
    batch = [((3,H,W), label)]*batch_size
    """
    
    data = torch.stack([b[0] for b in batch], dim = 0)
    label = torch.tensor([b[1] for b in batch])
    
    return data,label