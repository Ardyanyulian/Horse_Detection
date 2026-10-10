from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# 1. Siapkan data penampung untuk Validation
path_horses_val = './dataset/validation/horses/'
path_human_val = './dataset/validation/humans/'
model_terlatih = joblib.load('model_horse_human_linear.joblib')
X_val = []
y_val = []

# Load gambar Kuda di folder validation
for file in os.listdir(path_horses_val):
  img_path = os.path.join(path_horses_val, file)
  img = Image.open(img_path).convert('L').resize((128, 128))
  X_val.append(np.array(img).flatten())
  y_val.append(0)  # Label Kuda

# Load gambar Manusia di folder validation
for file in os.listdir(path_human_val):
  img_path = os.path.join(path_human_val, file)
  img = Image.open(img_path).convert('L').resize((128, 128))
  X_val.append(np.array(img).flatten())
  y_val.append(1)  # Label Manusia

# 2. Prediksi Semua Data Validation menggunakan model yang sudah di-load
y_pred = model_terlatih.predict(X_val)


cm = confusion_matrix(y_val, y_pred)

print("==           Confusion Matrix             ==")
print(" Condition  | Predict horses | Predict Human")
print(f"True Horses: {cm[0][0]:<12}   {cm[0][1]}")
print(f"True Human : {cm[1][0]:<12}   {cm[1][1]}")


print("== Classification Report ==")
print(
        classification_report(
            y_val, y_pred, target_names=['Horses (0)', 'Human (1)']
            )
        )
