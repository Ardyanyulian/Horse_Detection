🐴 vs 🧍 Horse & Human Image Classification (Classic ML)

Eksperimen klasifikasi citra digital untuk membedakan gambar **Kuda** (*Horse*) dan **Manusia** (*Human*) menggunakan pendekatan **Machine Learning Klasik** tanpa Deep Learning / CNN.

Eksperimen ini bertujuan untuk menguji batas kemampuan serta perbedaan performa antara algoritma berbasis fungsi jarak (SVM dengan RBF Kernel) dan algoritma berbasis keputusan hirarki (Random Forest) ketika menangani data citra 2D yang di-flatten menjadi vektor 1D.

---

## 📊 Hasil Eksperimen & Perbandingan Model

| Algoritma | Kernel / Parameter | Akurasi Test Set | Catatan Performa |
| :--- | :--- | :---: | :--- |
| **SVM (Support Vector Machine)** | **RBF (Radial Basis Function)** | **91%** | 🏆 **Model Terbaik**. Berhasil memetakan batas keputusan non-linear citra dengan akurat. |
| **SVM (Support Vector Machine)** | Linear | ~80% | Kurang optimal karena fitur piksel gambar bersifat non-linear kompleks. |
| **Random Forest** | Default Trees | **67%** | ⚠️ **Performa Buruk**. Mengalami penurunan signifikan akibat efek *flattening* 2D ke 1D. |

---

## 🔍 Analisis Hasil & Pembelajaran Teknis

1. **Mengapa SVM RBF Unggul (91%)?**
   * Kernel RBF (*Radial Basis Function*) mampu memetakan vektor fitur piksel ke dalam ruang dimensi tinggi (*hyperplane* non-linear).
   * SVM efektif menangani data berdimensi tinggi (*high-dimensional space*) meskipun struktur piksel di-flatten.

2. **Mengapa Random Forest Gagal (67%)?**
   * **Kehilangan Struktur Spasial (2D -> 1D)**: Proses pemadatan matriks piksel 2D menjadi array 1D menghancurkan hubungan keterdekatan antar-piksel (*spatial relationship*).
   * *Decision Tree* membagi data berdasarkan ambang batas piksel individual (*axis-aligned splits*), yang sangat tidak efektif untuk mengenali pola visual terdistribusi seperti bentuk tubuh kuda atau manusia.

---

## 📁 Struktur Repository

```text
.
├── main.py                          # Training pipeline untuk SVM (Model Terbaik)
├── mainRandomForest.py              # Training pipeline untuk Random Forest
├── mainTuning64Linear.py            # Eksperimen tuning SVM Linear (64x64)
├── cobaAkurasi.py                   # Script evaluasi akurasi SVM
├── cobaAkurasiRandomForest.py       # Script evaluasi akurasi Random Forest
├── cobaConfusionMatrik.py           # Evaluation matrix & visualization
├── cobaModel.py                     # Script inference / pengujian gambar baru
└── .gitignore                       # Mengecualikan model (.joblib) & dataset
