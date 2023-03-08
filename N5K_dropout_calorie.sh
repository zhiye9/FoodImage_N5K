#!/bin/bash 
module load tools
#module load torchmetrics/0.11.1
module load anaconda3/2022.10
module load cuda/toolkit/11.8.0
module list
python3 /home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/FoodImage_N5K/n5k_model_dropout.py -a 'swin_b' --train_path "/home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/FoodImage_N5K/data_id/df_train_all_id_calories.txt" --test_path "/home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/FoodImage_N5K/data_id/df_test_all_id_calories.txt" --output_model '/home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/FoodImage_N5K/results/checkpoint_150_lr1e-4_calories_swin_b.pth' --output_val /home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/FoodImage_N5K/results/hist_150_val_lr1e-4_calories_swin_b --output_train /home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/FoodImage_N5K/results/hist_150_train_lr1e-4_calories_swin_b --epoch 150



 
