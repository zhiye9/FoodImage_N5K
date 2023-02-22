#!/bin/bash 
module load tools
module load pytorch/1.12.1 
module load anaconda3/2020.07
python3 /home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/n5k_model.py -a resnet50 --output_model './checkpoint_2_lr001_res152.pth' --output_val hist_5_val_lr1e-4_res152 --output_train hist_5_train_lr1e-4_res152
