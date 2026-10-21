# Data Scientist Retention Model 🚀

Repositori ini berisi implementasi **Machine Learning Pipeline** untuk menganalisis dan memprediksi tingkat perpindahan karyawan (*turnover/flight risk*) pada talenta Data Science. Proyek ini dikembangkan untuk kebutuhan HR Analytics.

---

## 📦 Prasyarat

- Python **3.12+**
- pip (sudah termasuk dalam Python)
- Git

---

## ⚙️ Instalasi & Setup

### 1. Clone Repository

```bash
git clone https://github.com/fachridtya/data-scientist-retention-model.git
cd data-scientist-retention-model
```

### 2. Buat & Aktifkan Virtual Environment

> **Mengapa virtual environment?**
> Virtual environment mengisolasi dependensi proyek dari sistem utama, mencegah konflik antar proyek.

**Linux / macOS / WSL:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> ✅ Setelah aktif, terminal akan menampilkan `(.venv)` di awal prompt.

### 3. Install Dependensi

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Siapkan Dataset

Pastikan file dataset `aug_train.csv` berada di folder:

```
data/raw/aug_train.csv
```

> Dataset asli bersumber dari [Kaggle HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists).

---

## 🚀 Menjalankan Pipeline

Jalankan seluruh pipeline (Preprocessing → Modeling → Evaluasi) dengan satu perintah:

```bash
python3 main.py
```

Atau jalankan setiap tahap secara terpisah:

| Tahap            | Perintah                                     |
| ---------------- | -------------------------------------------- |
| 1. Preprocessing | `python3 src/preparation/preprocessing.py` |
| 2. Split Data    | `python3 src/preparation/splitter.py`      |
| 3. Transformasi  | (dipanggil dari`main.py`)                  |
| 4. Seleksi Fitur | (dipanggil dari`main.py`)                  |
| 5. Training DT   | `python3 src/modeling/decision_tree/train.py`   |
| 6. Evaluasi DT   | `python3 src/modeling/decision_tree/evaluate.py` |
| 7. Training SVM  | `python3 src/modeling/svm/train.py`             |
| 8. Evaluasi SVM  | `python3 src/modeling/svm/evaluate.py`          |
| 9. Training NB   | `python3 src/modeling/naive_bayes/train.py`     |
| 10. Evaluasi NB  | `python3 src/modeling/naive_bayes/evaluate.py`  |

> **Catatan**: Tahap 3–4 membutuhkan output dari tahap sebelumnya, jadi jalankan melalui `main.py`. Tahap 5–10 juga bergantung pada `data/processed/` yang dihasilkan tahap 4.

---

## 🖥️ Menjalankan Aplikasi Web (Deployment)

Setelah pipeline selesai dan artefak model tersedia di `assets/`, jalankan aplikasi inferensi:

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`. Anda dapat mengisi profil kandidat melalui formulir dan menekan tombol **Analyze Candidate ✨** untuk melihat hasil prediksi.

> **Catatan**: `app.py` memuat `assets/model_svm.pkl` — yaitu model dengan kinerja terbaik hasil perbandingan pada `main.py`. Pastikan pipeline sudah dijalankan minimal sekali sebelum membuka aplikasi.

### Deaktivasi Virtual Environment

Setelah selesai, nonaktifkan virtual environment:

```bash
deactivate
```

---

## 🎯 Business Objective

Membantu tim Human Resources (HR) mengidentifikasi kandidat atau karyawan yang memiliki probabilitas tinggi untuk *resign* atau mencari pekerjaan baru (Target = 1), sehingga perusahaan dapat menerapkan strategi retensi yang proaktif dan menyelamatkan aset SDM terbaiknya.

---

## 🛠️ Pipeline & Metodologi

Proyek ini dikerjakan melalui pendekatan terstruktur yang dibagi menjadi dua fase utama:

### Fase 1: Data Engineering & Feature Selection

| Tahap                   | Keterangan                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------ |
| **Preprocessing** | Deteksi & handling duplikat, missing value (Modus/Median), dan outlier (IQR Capping) |
| **Encoding**      | Konversi data kategorikal ke numerik menggunakan LabelEncoder                        |
| **Split Data**    | Pemisahan 80% Training, 20% Testing (`random_state=42`)                              |
| **Transformasi**  | Normalisasi fitur numerik menggunakan Min-Max Scaling                                |
| **Seleksi Fitur** | Uji Chi-Square (kategorikal) & T-test (numerik) untuk memilih fitur signifikan       |

### Fase 2: Modeling & Evaluation

| Tahap | Keterangan |
| ------------------ | ---------------------------------------------------- |
| **Modeling** | Klasifikasi komparatif 3 algoritma: Decision Tree, Support Vector Machine (SVM), dan Gaussian Naïve Bayes |
| **Evaluasi** | Confusion Matrix, Akurasi, Presisi, Recall, F1-Score untuk setiap model |
| **Deployment** | Aplikasi web inferensi (`app.py`) berbasis Streamlit dengan tema UI Glassmorphism |

---

## 📂 Struktur Proyek

Hanya berkas yang di-*track* Git yang ditampilkan di bawah ini:

```
.
├── .gitignore               # Aturan file yang tidak di-track Git
├── README.md                # Dokumentasi proyek
├── requirements.txt         # Daftar dependensi Python
├── main.py                  # Entry point pipeline end-to-end
├── app.py                   # Aplikasi Streamlit (deployment model terbaik)
├── assets/                  # Model terlatih & artefak visualisasi
│   ├── model.pkl                    # Decision Tree terlatih
│   ├── model_svm.pkl                # SVM terlatih (= model yang di-deploy)
│   ├── naive_bayes_model.pkl        # Naïve Bayes terlatih
│   ├── scaler.pkl                   # MinMaxScaler terlatih
│   ├── encoders.pkl                 # LabelEncoder per kolom
│   ├── selected_features.json       # Daftar 9 fitur terpilih
│   ├── confusion_matrix.png         # Confusion Matrix Decision Tree
│   ├── confusion_matrix_svm.png     # Confusion Matrix SVM
│   ├── confusion_matrix_nb.png      # Confusion Matrix Naïve Bayes
│   ├── decision_tree_depth3.png     # Visualisasi pohon (depth 3)
│   ├── decision_tree_depth5.png     # Visualisasi pohon (depth 5)
│   └── feature_importance.png       # Feature importance Decision Tree
├── data/
│   └── raw/                 # Dataset mentah (di-track)
│       └── aug_train.csv
└── src/
    ├── preparation/
    │   ├── preprocessing.py     # Duplikat, missing value, outlier (IQR Capping)
    │   ├── splitter.py          # Label Encoding & split 80/20
    │   ├── transformation.py    # Min-Max Scaling
    │   └── selection.py         # Seleksi fitur (Chi-Square & T-test)
    └── modeling/
        ├── decision_tree/
        │   ├── train.py         # Training Decision Tree
        │   └── evaluate.py      # Evaluasi Decision Tree
        ├── svm/
        │   ├── train.py         # Training SVM
        │   └── evaluate.py      # Evaluasi SVM
        └── naive_bayes/
            ├── train.py         # Training Naïve Bayes
            └── evaluate.py      # Evaluasi Naïve Bayes
```

> **Catatan**: Direktori `data/interim/`, `data/processed/`, dan `.venv/` tidak ditampilkan karena di-*ignore* oleh `.gitignore`. Ketiga direktori `data/` tersebut dibuat otomatis saat `main.py` dijalankan. Artefak di `assets/` di-*track* Git agar model dapat langsung dipakai untuk inferensi melalui `app.py` tanpa perlu melatih ulang.

---

## 💻 Tech Stack

| Teknologi                      | Kegunaan                                          |
| ------------------------------ | ------------------------------------------------- |
| **Python 3.12+**         | Bahasa pemrograman utama                          |
| **Pandas & NumPy**       | Manipulasi & analisis data                        |
| **Scikit-Learn**         | Machine Learning (preprocessing, model, evaluasi) |
| **SciPy**                | Uji statistik (Chi-Square, T-test)                |
| **Matplotlib & Seaborn** | Visualisasi Confusion Matrix                      |
| **Streamlit**            | Deployment model sebagai aplikasi web interaktif  |
| **imbalanced-learn**     | Resampling untuk data tidak seimbang (opsional)   |

---

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan akademik — Program Studi D4 Teknik Informatika, Universitas Airlangga.
