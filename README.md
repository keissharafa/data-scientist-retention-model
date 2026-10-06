# Data Scientist Retention Model 🚀

Repositori ini berisi implementasi *Machine Learning Pipeline* untuk menganalisis dan memprediksi tingkat perpindahan karyawan (*turnover/flight risk*) pada talenta Data Science. Proyek ini dikembangkan untuk kebutuhan HR Analytics.

## 🎯 Business Objective
Membantu tim Human Resources (HR) mengidentifikasi kandidat atau karyawan yang memiliki probabilitas tinggi untuk *resign* atau mencari pekerjaan baru (Target = 1), sehingga perusahaan dapat menerapkan strategi retensi yang proaktif dan menyelamatkan aset SDM terbaiknya.

## 🛠️ Pipeline & Metodologi
Proyek ini dikerjakan melalui pendekatan kolaboratif terstruktur yang dibagi menjadi dua fase utama:

**Fase 1: Data Engineering & Feature Selection**
* Pembersihan data (*Handling Missing Values* dengan Modus).
* Penanganan anomali data (*Outlier Capping* menggunakan IQR/Boxplot).
* Pemisahan dataset (80% Training, 20% Testing) untuk mencegah *data leakage*.
* Transformasi rentang nilai menggunakan **Min-Max Scaling**.
* Seleksi fitur statistik independen menggunakan **Chi-Square** (kategorikal) dan **T-test** (numerik).

**Fase 2: Modeling & Evaluation**
* Pelatihan model klasifikasi menggunakan algoritma **Decision Tree**.
* Evaluasi performa algoritma menggunakan *Confusion Matrix* (Akurasi, Presisi, Recall, dan F1-Score).

## 💻 Tech Stack
* Python
* Pandas & NumPy
* Scikit-Learn
* SciPy
