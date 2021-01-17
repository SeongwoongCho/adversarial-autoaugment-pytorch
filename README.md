# Adversarial-Autoaugment-Pytorch
Unofficial Pytorch Implementation Of [AdversarialAutoAugment(ICLR2020)](https://arxiv.org/pdf/1912.11188.pdf)

## Quick Start
```
# Training with DistributedDataParallel
$ python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} main.py \
    --load_conf ${conf_dir} \
    --logdir './logs'
    --M 8

# Training with DataParallel
$ python main.py \
    --load_conf ${conf_dir} \
    --logdir './logs'
    --M 8

# Evaluate
$ python evaluate.py \
    --load_conf ${conf_dir} \
    --logdir './logs'

# RUN EXPS
python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar10/wresnet28x10_cifar10.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar10/shake26_2x32d_cifar10.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar10/shake26_2x96d_cifar10.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar100/wresnet28x10_cifar100.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar100/shake26_2x96d_cifar100.yaml' --logdir './logs/' --M 8 --amp >> output.log
```

## Results
| Model(CIFAR-10)         |   Paper   |  Ours     |    |
|-------------------------|-----------|-----------|----|
| Wide-ResNet-28-10       | 1.90±0.15 |     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Shake-Shake(26 2x32d)   | 2.36±0.10 |     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Shake-Shake(26 2x96d)   | 1.85±0.12 |     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Shake-Shake(26 2x112d)  | 1.78±0.05 |     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| PyramidNet+ShakeDrop    | 1.36±0.06 |     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |

| Model(CIFAR-100)        |   Paper   |   Ours    |    |
|-------------------------|-----------|-----------|----|
| Wide-ResNet-28-10       | 15.49±0.18|     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Shake-Shake(26 2x96d)   | 14.10±0.15|     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| PyramidNet+ShakeDrop    | 10.42±0.20|     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |

| Model(ImageNet)         |          Paper         |           Ours          |    |
|-------------------------|------------------------|-------------------------|----|
| Resnet50                | 20.60±0.15 / 5.53±0.05 |            -            |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Resnet50D               | 20.00±0.12 / 5.25±0.03 |            -            |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Resnet200               | 18.68±0.18 / 4.70±0.05 |            -            |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |

## Different From The Paper
- I did not include SamplePairing -> NUM_OPS = 15 (16 in the paper)
- Model optimizer : SGD로 설정해놓음

## TODO

1. controller 이슈 -> 샘플링 방법 
2. valid loader 대신 test loader를 원래 바로쓰나..? split size? 
3. entropy penalty 부호는 음수가 맞겠지? 

Tensorboard

현재 상태 - minor error 고친상태.
표에 Transfer추가하기.

__pycache__
.ipynb_checkpoints 삭제 후 ignore하기

CIFAR-10-C, CIFAR-10-P 테스트 코드 추가하기 

## Observations
- Optimizer를 SGD -> SGD momentum으로 바꿔서 학습을 하니깐 더 잘됨. controller의 스케쥴링등에 영향이 있을듯? 
- controller가 얼마나 민감한가? Controller vs Network 간의 조절이 중요할듯 

- controller ver 1(hidden state계속 subpolicy마다 forward할 때) -> 후반부로 갈수록 엄청 강한 augmentation이 나온다. (translateY,0)같은 것들
=> 따라서 후반부로 갈수록 training loss가 엄청 높아지고, 논문 결과가 재현되지는 않는다. But 그래도 AA보다는 좋은 성능을 보인다. 
=> 이는 testloader를 validloader로 활용하지 않을 때 결과이고, 만약 testloader로 validating을 한다면 논문에 더 근접한 성능을 보여주지 않을까?  

- controller ver 2(hidden state를 subpolicy별로 init할 때) ->


## References & Open Sources
ENAS(https://github.com/carpedm20/ENAS-pytorch)

FastAutoAugment(https://github.com/kakaobrain/fast-autoaugment)
