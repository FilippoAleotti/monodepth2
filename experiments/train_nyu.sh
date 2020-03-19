#!bin/bash

#arch="pydnet"
arch="resnet"
#arch="mobilenet"
extra=""
logs="./logs"
dataset="/media/filippo/nvme/ComputerVision/Dataset/NYUv2"
#dataset="/media/faleotti/SSD/NYUv2"
stereo=1

if [[ $arch == 'mobilenet' ]];
then
    extra="$extra --scales 0"
fi

if [[ $arch == 'pydnet' ]];
then
    extra="$extra  --scales 1 2 3"
fi

python train.py --architecture $arch   \
                --log_dir $logs \
                --data_path $dataset \
                --split nyu \
                --dataset nyu \
                --png \
                --width 320  --height 224 \
                $extra