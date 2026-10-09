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
| 5. Training      | `python3 src/modeling/train.py`            |
| 6. Evaluasi      | `python3 src/modeling/evaluate.py`         |

> **Catatan**: Tahap 3–4 membutuhkan output dari tahap sebelumnya, jadi jalankan melalui `main.py`.

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
| **Split Data**    | Pemisahan 80% Training, 20% Testing dengan stratifikasi                              |
| **Transformasi**  | Normalisasi fitur numerik menggunakan Min-Max Scaling                                |
| **Seleksi Fitur** | Uji Chi-Square (kategorikal) & T-test (numerik) untuk memilih fitur signifikan       |

### Fase 2: Modeling & Evaluation

| Tahap              | Keterangan                                           |
| ------------------ | ---------------------------------------------------- |
| **Modeling** | Klasifikasi menggunakan Decision Tree Classifier     |
| **Evaluasi** | Confusion Matrix, Akurasi, Presisi, Recall, F1-Score |

---

## 📂 Struktur Proyek

```
.
├── .gitignore               # Aturan file yang tidak di-track Git
├── .venv/                    # Virtual environment (tidak di-track)
├── requirements.txt         # Daftar dependensi Python
├── README.md
├── main.py                  # Entry point pipeline end-to-end
├── assets/                  # Model & artifacts (tidak di-track)
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── selected_features.json
│   └── confusion_matrix.png
├── data/
│   ├── raw/                 # Dataset mentah (di-track)
│   │   └── aug_train.csv
│   ├── interim/             # Data setelah preprocessing (tidak di-track)
│   └── processed/           # Data siap modeling (tidak di-track)
├── src/
│   ├── preparation/
│   │   ├── preprocessing.py
│   │   ├── splitter.py
│   │   ├── transformation.py
│   │   └── selection.py
│   └── modeling/
│       ├── train.py
│       └── evaluate.py
```

---

## 💻 Tech Stack

| Teknologi                      | Kegunaan                                          |
| ------------------------------ | ------------------------------------------------- |
| **Python 3.12+**         | Bahasa pemrograman utama                          |
| **Pandas & NumPy**       | Manipulasi & analisis data                        |
| **Scikit-Learn**         | Machine Learning (preprocessing, model, evaluasi) |
| **SciPy**                | Uji statistik (Chi-Square, T-test)                |
| **Matplotlib & Seaborn** | Visualisasi Confusion Matrix                      |
| **imbalanced-learn**     | Resampling untuk data tidak seimbang              |

---

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan akademik — Program Studi D4 Teknik Informatika, Universitas Airlangga.
