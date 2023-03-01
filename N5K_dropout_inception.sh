#!/bin/bash 
module load tools
#module load torchmetrics/0.11.1
module load anaconda3/2022.10
module load cuda/toolkit/11.8.0
module list
python3 /home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/FoodImage_N5K/n5k_model_dropout.py -a 'inception_v3' --output_model '/home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/FoodImage_N5K/results/checkpoint_150_dropout_lr1e-4_inception_v3.pth' --output_val /home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/FoodImage_N5K/results/hist_150_dropout_val_lr1e-4_inception_v3 --output_train /home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/FoodImage_N5K/results/hist_150_dropout_train_lr1e-4_inception_v3 --epoch 150 --is_inception True
