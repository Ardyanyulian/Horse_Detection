from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image
import os
import joblib

# Variabel penampung
path_horses_train = "./dataset/train/horses/"
path_human_train = "./dataset/train/humans/"

array_path_horses = []
array_path_human = []

# main variabel
X  = []
Y = []

# Insertion Path
for file in os.listdir(path_horses_train):
    path_complete_horse = os.path.join(path_horses_train, file)
    array_path_horses.append(path_complete_horse)

for file in os.listdir(path_human_train):
    path_complete_human = os.path.join(path_human_train, file)
    array_path_human.append(path_complete_human)


for img_horse in array_path_horses:
    img_raw_horse = Image.open(img_horse)
    img_modified_horse = img_raw_horse.convert('L').resize((128,128))
    img_array = np.array(img_modified_horse)
    pixel_1d = img_array.flatten()
    X.append(pixel_1d)
    Y.append(0)

for img_human in array_path_human:
    img_raw_human = Image.open(img_human)
    img_modified_human = img_raw_human.convert('L').resize((128,128))
    img_array = np.array(img_modified_human)
    pixel_1d = img_array.flatten()
    X.append(pixel_1d)
    Y.append(1)

clf = make_pipeline(StandardScaler(), svm.SVC(kernel="linear", cache_size=1000))

clf.fit(X,Y)
joblib.dump(clf, "model_horse_human_linear.joblib")

model_terlatih = joblib.load('model_horse_human_linear.joblib')
