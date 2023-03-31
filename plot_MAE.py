import pickle
import matplotlib.pyplot as plt

PATH_train = "/home/zhi/data/FoodImage_N5K/results/hist_150_train_lr1e-4_calories_swin_b"
PATH_test = "/home/zhi/data/FoodImage_N5K/results/hist_150_val_lr1e-4_calories_swin_b"

with open(PATH_train, "rb") as fp:
    hist_50 = pickle.load(fp)

with open(PATH_test, "rb") as fp:
    hist_50_lr0001 = pickle.load(fp)

hist1 = []
hist1 = [h for h in hist_50]
hist2 = []
hist2 = [h for h in hist_50_lr0001]
num_epochs = 150

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

plt.title("Swin-transformer for fat")
plt.xlabel("Training Epochs")
plt.ylabel("MAE")
plt.plot(range(1,num_epochs+1),hist1,label="Training MAE (150 Epochs with lr 1e-4)")
plt.plot(range(1,num_epochs+1),hist2,label="Validation MAE (150 Epochs with lr 1e-4)")
#plt.yticks([])
plt.legend()
plt.show()

