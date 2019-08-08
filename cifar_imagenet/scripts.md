python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adam --lr 0.001 --beta1 0.9 --beta2 0.999 --checkpoint checkpoints/cifar10/resnet-20-adam --gpu-id 1

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-20-sgd --gpu-id 2

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer ada2 --lr 0.001 --beta1 0.9 --beta2 0.99 --checkpoint checkpoints/cifar10/resnet-20-ada --gpu-id 0
