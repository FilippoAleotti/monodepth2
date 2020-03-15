#!bin/bash

arch="pydnet"
extra=""
logs="./logs"
#kitti="/media/filippo/nvme/ComputerVision/Dataset/FULL_KITTI"
kitti="/media/faleotti/SSD/FULL_KITTI"
stereo=1

if [[ $stereo -eq 1 ]]; 
then
    extra="$extra --use_stereo"
fi

if [[ $arch == 'pydnet' ]];
then
    extra="$extra --scales 1 2 3"
fi
python train.py --architecture $arch --frame_ids 0  \
                --split eigen_full $extra --log_dir $logs \
                --data_path $kitti