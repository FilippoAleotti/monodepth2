#!bin/bash

#arch="pydnet"
arch="resnet"
#arch="mobilenet"
extra=""
logs="./logs"
dataset="/media/filippo/nvme/ComputerVision/Dataset/DIODE"
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
                --split diode \
                --dataset diode \
                --png \
                --width 640  --height 448 \
                $extra