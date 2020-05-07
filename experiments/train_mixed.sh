#!bin/bash

#arch="pydnet"
#arch="resnet"
arch="mobilenet"
extra=""
logs="./logs"
dataset="/media/faleotti/SSD/MONO_WILD"

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
                --split mixed \
                --dataset mixed \
                --png \
                --width 640  --height 320 \
                $extra

