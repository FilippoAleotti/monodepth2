#!bin/bash

#arch="pydnet"
arch="resnet"
#arch="mobilenet"
extra=""
logs="./logs"
kitti="/media/filippo/nvme/ComputerVision/Dataset/FULL_KITTI"
#kitti="/media/faleotti/SSD/FULL_KITTI"
stereo=1
split="eigen"

if [[ $stereo -eq 1 ]]; 
then
    extra="$extra --use_stereo"
fi

if [[ $arch == "resnet" ]];
then
    ckpt="logs/resnet"
    scale=0
fi

if [[ $arch == "pydnet" ]];
then
    ckpt="logs/pydnet/weights_49"
    scale=1
fi


if [ ! -f "splits/$split/gt_depths.npz" ]; 
then
    echo "Creating depth.npz file in splits/$split/"
    python export_gt_depth.py --data_path $kitti --split $split
fi

echo "Running evaluation on Eigen split (standard)"

python evaluate_depth.py --load_weights_folder $ckpt --eval_stereo --architecture $arch --data_path $kitti --prediction_scale $scale
