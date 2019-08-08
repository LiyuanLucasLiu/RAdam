
python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-radam-01 --gpu-id 1 --model_name radam_01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-radam-003 --gpu-id 2 --model_name radam_003 --lr 0.03

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-radam-001 --gpu-id 3 --model_name radam_001 --lr 0.01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-radam-0003 --gpu-id 2 --model_name radam_0003 --lr 0.003

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-radam-0001 --gpu-id 3 --model_name radam_0001 --lr 0.001
