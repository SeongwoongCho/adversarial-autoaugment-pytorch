# Adversarial-Autoaugment-Pytorch(ICLR2020)
Unofficial Pytorch Implementation Of AdversarialAutoAugment(ICLR2020)

# TODO
- tensorboard

# Quick Start
```
# run train.py with DDP
$ python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} main.py \
    --load_conf ${conf_dir} \
    --logdir './logs'
    --M 8

# run train.py with DP
$ python main.py \
    --load_conf './confs/cifar10/wresnet28x10_cifar10.yaml' \
    --logdir './logs'
    --M 8

# RUN EXPS
python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar10/wresnet28x10_cifar10.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar10/shake26_2x32d_cifar10.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar10/shake26_2x96d_cifar10.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar100/wresnet28x10_cifar100.yaml' --logdir './logs/' --M 8 --amp >> output.log && python -m torch.distributed.launch --nproc_per_node 2 main.py --load_conf './confs/cifar100/shake26_2x96d_cifar100.yaml' --logdir './logs/' --M 8 --amp >> output.log

```

# Different from The Paper
- I did not include SamplePairing -> NUM_OPS = 15 (16 in the paper)
- Model optimizer : SGD로 설정해놓음

# References
ENAS(https://github.com/carpedm20/ENAS-pytorch)

fast-autoaugment(https://github.com/kakaobrain/fast-autoaugment)
