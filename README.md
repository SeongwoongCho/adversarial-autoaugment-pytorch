# Adversarial-Autoaugment-Pytorch
Unofficial Pytorch Implementation Of [AdversarialAutoAugment(ICLR2020)](https://arxiv.org/pdf/1912.11188.pdf)

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
| Shake-Shake(26 2x96d)   | 1.85±0.12 / -        |     -     |[Download](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch) |
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
- Model optimizer : fast-autoaugment를 참고

## TODO

CIFAR-10-C auto-download

CIFAR-10-C, CIFAR-10-P 테스트 코드 추가하기 

## Observations
- Optimizer를 뭘로 해야하는가?
- controller가 얼마나 민감한가? Controller vs Network 간의 조절이 중요할듯 

- controller ver 1(hidden state계속 subpolicy마다 forward할 때) -> 후반부로 갈수록 엄청 강한 augmentation이 나온다. (translateY,0)같은 것들
=> 따라서 후반부로 갈수록 training loss가 엄청 높아지고, 논문 결과가 재현되지는 않는다. But 그래도 AA보다는 좋은 성능을 보인다. 
=> 이는 testloader를 validloader로 활용하지 않을 때 결과이고, 만약 testloader로 validating을 한다면 논문에 더 근접한 성능을 보여주지 않을까?
=> collapse of adversarial training? (learning rate, entropy penalty)

## References & Open Sources
[ENAS](https://github.com/carpedm20/ENAS-pytorch)

[FastAutoAugment](https://github.com/kakaobrain/fast-autoaugment)

[Augmix](https://github.com/google-research/augmix)

[OpenReview of AdvAA](https://openreview.net/forum?id=ByxdUySKvS)
