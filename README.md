# Adversarial-Autoaugment-Pytorch
Unofficial Pytorch Implementation Of [AdversarialAutoAugment(ICLR2020)](https://arxiv.org/pdf/1912.11188.pdf)

## Current Issue
I want some help from those who know how to solve these issues. 

- Can not reproduce paper's results

- Adversarial Collapsing : See /Examples/Analysis.ipynb



## Quick Start
```
# Training with DistributedDataParallel
$ python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} main.py \
    --load_conf ${conf_dir} \
    --logdir './logs' \
    --M 8 \ 
    --seed 0 \
    -- amp \ 
    >> output.log

# Training with DataParallel
$ python main.py \
    --load_conf ${conf_dir} \
    --logdir './logs' \
    --M 8 \ 
    --seed 0 \
    -- amp \
    >> output.log

# Evaluate
$ python evaluate.py \
    --load_conf ${conf_dir} \
    --logdir './logs' \
    --seed 0 
```

## Results
### CIFAR-10

| Model(CIFAR-10)         |Paper<br/>(direct/transfer)|  Ours     |    |
|-------------------------|----------------------|-----------|----|
| Wide-ResNet-28-10       | 1.90±0.15 / 2.45±0.13| 2.35 / -  |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Shake-Shake(26 2x32d)   | 2.36±0.10 / -        | 2.51 / -  |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Shake-Shake(26 2x96d)   | 1.85±0.12 / -        | 2.43 / -  |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Shake-Shake(26 2x112d)  | 1.78±0.05 / -        |     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| PyramidNet+ShakeDrop    | 1.36±0.06 / -        |     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |

### CIFAR-100

| Model(CIFAR-100)        |Paper<br/>(direct/transfer)|   Ours    |    |
|-------------------------|------------------------|-----------|----|
| Wide-ResNet-28-10       | 15.49±0.18 / 16.48±0.15|     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Shake-Shake(26 2x96d)   | 14.10±0.15 / -         |     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| PyramidNet+ShakeDrop    | 10.42±0.20 / -         |     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |

### ImageNet

| Model(ImageNet)         |Paper<br/>(top1/top5/ transfer_top1)|           Ours          |    |
|-------------------------|------------------------------------|-------------------------|----|
| Resnet50                | 20.60±0.15 / 5.53±0.05 / -         |            -            |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Resnet50D               | 20.00±0.12 / 5.25±0.03 / 20.20±0.05|            -            |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Resnet200               | 18.68±0.18 / 4.70±0.05 / 19.05±0.10|            -            |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |

### CIFAR-10-C

| Model(CIFAR-10-C)       |     Augmix w/ JSD  |          Adv AA         |    |
|-------------------------|--------------------|-------------------------|----|
| Wide-Resnet-40-2        |        11.2        |            -            |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Wide-Resnet-28-10       |         -          |          10.41          |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
| Shake-Shake(26 2x32d)   |         -          |          16.69          |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |

## Different From The Paper
- I did not include SamplePairing -> NUM_OPS = 15 (16 in the paper)
- Borrow unknown hyperparameter settings from fast-autoaugment

## TODO

## Observations

## References & Open Sources
[ENAS](https://github.com/carpedm20/ENAS-pytorch)

[FastAutoAugment](https://github.com/kakaobrain/fast-autoaugment)

[Augmix](https://github.com/google-research/augmix)

[OpenReview of AdvAA](https://openreview.net/forum?id=ByxdUySKvS)
