
ROOT="cps"
for SEED in 1111 2222 3333 4444 5555
do
	python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam4s  --beta1 0.9 --beta2 0.999  --checkpoint $ROOT/cifar10/resnet-20-adam4s-01-$SEED --gpu-id 0 --lr 0.1 --model_name adam4s --manualSeed $SEED

	python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam4s  --beta1 0.9 --beta2 0.999  --checkpoint $ROOT/cifar10/resnet-20-adam4s-ua-01-$SEED --gpu-id 0 --lr 0.1 --model_name adam4s_ua --update_all --manualSeed $SEED

	python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer radam4s  --beta1 0.9 --beta2 0.999  --checkpoint $ROOT/cifar10/resnet-20-adam4s-ua-af-01-$SEED --gpu-id 0 --lr 0.1 --model_name adam4s_ua_af --update_all --additional_four --manualSeed $SEED
done
