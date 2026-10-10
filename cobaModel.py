from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import numpy as np
from PIL import Image
import os
import joblib
import sklearn


# 1. Ambil 1 foto yang mau dites
file_tes = './dataset/validation/horses/horse1-000.png'
model_terlatih = joblib.load('model_horse_human.joblib')
# 2. Samakan proses pembacaannya (Resize 64x64, Grayscale, Flatten)
img = Image.open(file_tes).convert('L').resize((64, 64))
pixel_tes = np.array(img).flatten()

# 3. Predict (PENTING: bungkus dengan [pixel_tes] karena predict butuh array 2D)
hasil = model_terlatih.predict([pixel_tes])

# 4. Cek Hasilnya
if hasil[0] == 0:
  print('Tebakan Model: KUDA (Horse)')
else:
  print('Tebakan Model: MANUSIA (Human)')