#!bin/bash

#arch="pydnet"
arch="resnet"
#arch="mobilenet"
extra=""
logs="./logs"
#dataset="/media/filippo/nvme/ComputerVision/Dataset/DIODE"
dataset="/media/faleotti/Storage1/DIODE"
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
                --width 768  --height 576 \
                $extra

