import torch
import pickle
import pandas as pd
import os
from os import listdir
from matplotlib import image
from skimage.transform import resize
import numpy as np
#import cv2
import time
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

'''
df1 = pd.read_csv('/home/zhi/nas/food_img/Nutrition5k/metadata/dish_metadata_cafe1.csv', header = None, index_col = False, on_bad_lines='skip')
df1 = df1.iloc[:, :6]
df1.columns = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']
df2 = pd.read_csv('/home/zhi/nas/food_img/Nutrition5k/metadata/dish_metadata_cafe2.csv', header = None, index_col = False, on_bad_lines='skip')
df2 = df2.iloc[:, :6]
df2.columns = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']

df_all = pd.concat([df1, df2], ignore_index=True, axis=0)
df_all.to_csv('/home/zhi/nas/food_img/Nutrition_pred/nutrition_all.csv', index = False)
'''
path_name1 = '/home/zhi/nas/food_img/Nutrition5k/metadata/dish_metadata_cafe1.csv'
delimiter = ','
max_columns1 = max(open(path_name1, 'r'), key = lambda x: x.count(delimiter)).count(delimiter) + 1
df1 = pd.read_csv('/home/zhi/nas/food_img/Nutrition5k/metadata/dish_metadata_cafe1.csv', header = None, index_col = False, names = list(range(0,max_columns1)))
df1_meal = df1.iloc[:, :6]
df1_meal.columns = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']

path_name2 = '/home/zhi/nas/food_img/Nutrition5k/metadata/dish_metadata_cafe2.csv'
delimiter = ','
max_columns2 = max(open(path_name2, 'r'), key = lambda x: x.count(delimiter)).count(delimiter) + 1
df2 = pd.read_csv('/home/zhi/nas/food_img/Nutrition5k/metadata/dish_metadata_cafe2.csv', header = None, index_col = False, names = list(range(0,max_columns2)))
df2_meal = df2.iloc[:, :6]
df2_meal.columns = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']

df_all = pd.concat([df1, df2], ignore_index=True, axis=0)
df_all.to_csv('/home/zhi/data/FoodImage_N5K/data_id/nutrition_all.csv', index = False)
df_meal_all = pd.concat([df1_meal, df2_meal], ignore_index=True, axis=0)
df_meal_all.to_csv('/home/zhi/data/FoodImage_N5K/data_id/nutrition_meal_all.csv', index = False)

os.chdir('data_id')

df_meal_all = pd.read_csv('nutrition_meal_all.csv')

txt_file = open("/home/zhi/nas/food_img/Nutrition5k/dish_ids/splits/rgb_train_ids.txt", "r")
file_content = txt_file.read()
train_id = file_content.split()
txt_file.close()
df_train_id = df_meal_all[df_meal_all['dish_id'].isin(train_id)]
df_train_id.to_csv('df_train_id1.csv', index=False)

txt_file = open("/home/zhi/nas/food_img/Nutrition5k/dish_ids/splits/rgb_test_ids.txt", "r")
file_content = txt_file.read()
test_id = file_content.split()
txt_file.close()
df_test_id = df_meal_all[df_meal_all['dish_id'].isin(test_id)]
df_test_id.to_csv('df_test_id.csv', index=False)

all_img = pd.read_csv('all_img.txt', header = None)
all_img.columns = ['file']
all_img = all_img.drop_duplicates(subset='file', keep="first")
#all_img_scp = all_img['file']
all_img['file'] = all_img['file'].str[2:]
all_img['dish_id'] = all_img['file'].str[:15]

df_train_all_id = pd.merge(df_train_id, all_img, on = 'dish_id')
df_train_all_id.to_csv('df_train_all_id.txt', index=False, header = False)
df_test_all_id = pd.merge(df_test_id, all_img, on = 'dish_id')
df_test_all_id.to_csv('df_test_all_id.txt', index=False, header = False)

df_train_all_id_carbs = df_train_all_id[['file', 'total_carb']]
df_train_all_id_carbs.to_csv('df_train_all_id_carbs.txt', index=False, header = False)
df_test_all_id_carbs = df_test_all_id[['file', 'total_carb']]
df_test_all_id_carbs.to_csv('df_test_all_id_carbs.txt', index=False, header = False)

df_train_all_id_calories = df_train_all_id[['file', 'total_calories']]
df_train_all_id_calories.to_csv('df_train_all_id_calories.txt', index=False, header = False)
df_test_all_id_calories = df_test_all_id[['file', 'total_calories']]
df_test_all_id_calories.to_csv('df_test_all_id_calories.txt', index=False, header = False)

df_train_all_id_mass = df_train_all_id[['file', 'total_mass']]
df_train_all_id_mass.to_csv('df_train_all_id_mass.txt', index=False, header = False)
df_test_all_id_mass = df_test_all_id[['file', 'total_mass']]
df_test_all_id_mass.to_csv('df_test_all_id_mass.txt', index=False, header = False)

df_train_all_id_fat = df_train_all_id[['file', 'total_fat']]
df_train_all_id_fat.to_csv('df_train_all_id_fat.txt', index=False, header = False)
df_test_all_id_fat = df_test_all_id[['file', 'total_fat']]
df_test_all_id_fat.to_csv('df_test_all_id_fat.txt', index=False, header = False)

df_train_all_id_protein	 = df_train_all_id[['file', 'total_protein']]
df_train_all_id_protein	.to_csv('df_train_all_id_protein.txt', index=False, header = False)
df_test_all_id_protein	 = df_test_all_id[['file', 'total_protein']]
df_test_all_id_protein	.to_csv('df_test_all_id_protein.txt', index=False, header = False)