# SGD 

```
python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-sgd-01 --gpu-id 0 --model_name sgd_01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-sgd-003 --gpu-id 0 --model_name sgd_003 --lr 0.03

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-sgd-001 --gpu-id 0 --model_name sgd_001 --lr 0.01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-sgd-0003 --gpu-id 0 --model_name sgd_0003 --lr 0.003
```

# Vanilla Adam

```
python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-adam-01 --gpu-id 0 --model_name adam_01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-adam-003 --gpu-id 0 --model_name adam_003 --lr 0.03

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-adam-001 --gpu-id 0 --model_name adam_001 --lr 0.01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-adam-0003 --gpu-id 0 --model_name adam_0003 --lr 0.003
```

# RAdam experiments

```
python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-radam-01 --gpu-id 0 --model_name radam_01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-radam-003 --gpu-id 0 --model_name radam_003 --lr 0.03

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-radam-001 --gpu-id 0 --model_name radam_001 --lr 0.01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-radam-0003 --gpu-id 0 --model_name radam_0003 --lr 0.003
```

# Adam with 100 warmup 

```
python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-01 --gpu-id 0 --warmup 100 --model_name adam_100_01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-003 --gpu-id 0 --warmup 100 --model_name adam_100_003 --lr 0.03

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-001 --gpu-id 0 --warmup 100 --model_name adam_100_001 --lr 0.01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-0003 --gpu-id 0 --warmup 100 --model_name adam_100_0003 --lr 0.003
```

# Adam with 200 warmup 

```
python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-01 --gpu-id 0 --warmup 200 --model_name adam_200_01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-003 --gpu-id 0 --warmup 200 --model_name adam_200_003 --lr 0.03

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-001 --gpu-id 0 --warmup 200 --model_name adam_200_001 --lr 0.01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-0003 --gpu-id 0 --warmup 200 --model_name adam_200_0003 --lr 0.003
```

# Adam with 500 warmup 

```
python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-01 --gpu-id 0 --warmup 500 --model_name adam_500_01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-003 --gpu-id 0 --warmup 500 --model_name adam_500_003 --lr 0.03

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-001 --gpu-id 0 --warmup 500 --model_name adam_500_001 --lr 0.01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-0003 --gpu-id 0 --warmup 500 --model_name adam_500_0003 --lr 0.003
```

# Adam with 1000 warmup 

```
python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-01 --gpu-id 0 --warmup 1000 --model_name adam_1000_01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-003 --gpu-id 0 --warmup 1000 --model_name adam_1000_003 --lr 0.03

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-001 --gpu-id 0 --warmup 1000 --model_name adam_1000_001 --lr 0.01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-adam-0003 --gpu-id 0 --warmup 1000 --model_name adam_1000_0003 --lr 0.003
```

# ImageNet

```
python imagenet.py -j 16 -a resnet18 --data /data/ILSVRC2012/ --epochs 90 --schedule 31 61 --gamma 0.1 -c ./cps/imagenet/resnet18_radam_0003 --model_name radam_0003 --optimizer radam --lr 0.003 --beta1 0.9 --beta2 0.999

python imagenet.py -j 16 -a resnet18 --data /data/ILSVRC2012/ --epochs 90 --schedule 31 61 --gamma 0.1 -c ./cps/imagenet/resnet18_radam_0005 --model_name radam_0005 --optimizer radam --lr 0.005 --beta1 0.9 --beta2 0.999
```
