from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image
import os
import joblib
from sklearn.metrics import accuracy_score


# 1. Siapkan data penampung untuk Validation
path_horses_val = './dataset/validation/horses/'
path_human_val = './dataset/validation/humans/'
model_terlatih = joblib.load("model_horse_human_randomforest.joblib")
X_val = []
y_val = []

# Load gambar Kuda di folder validation
for file in os.listdir(path_horses_val):
  img_path = os.path.join(path_horses_val, file)
  img = Image.open(img_path).convert('L').resize((64, 64))
  X_val.append(np.array(img).flatten())
  y_val.append(0)  # Label Kuda

# Load gambar Manusia di folder validation
for file in os.listdir(path_human_val):
  img_path = os.path.join(path_human_val, file)
  img = Image.open(img_path).convert('L').resize((64, 64))
  X_val.append(np.array(img).flatten())
  y_val.append(1)  # Label Manusia

y_pred = model_terlatih.predict(X_val)

accuration = accuracy_score(y_val, y_pred) * 100
print(f'Akurasi Model pada Data Validation : {accuration:.2f}%')
