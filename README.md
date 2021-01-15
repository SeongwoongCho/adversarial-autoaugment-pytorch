# Adversarial-Autoaugment-Pytorch
Unofficial Pytorch Implementation Of [AdversarialAutoAugment(ICLR2020)](https://arxiv.org/pdf/1912.11188.pdf)

## Quick Start
```
# run train.py with DDP
$ python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} main.py \
    --load_conf ${conf_dir} \
    --logdir './logs'
    --M 8

# run train.py with DP
$ python main.py \
    --load_conf ${conf_dir} \
    --logdir './logs'
    --M 8

# RUN EXPS
python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar10/wresnet28x10_cifar10.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar10/shake26_2x32d_cifar10.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar10/shake26_2x96d_cifar10.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar100/wresnet28x10_cifar100.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar100/shake26_2x96d_cifar100.yaml' --logdir './logs/' --M 8 --amp >> output.log
```

## Results
| Model(CIFAR-10)         |   Paper   |   Ours    |    |
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
Tensorboard

## References & Open Sources
ENAS(https://github.com/carpedm20/ENAS-pytorch)

FastAutoAugment(https://github.com/kakaobrain/fast-autoaugment)
