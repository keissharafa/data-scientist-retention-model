import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# STEP 1: INPUT DATA
# ==============================================================================
print("=== STEP 1: INPUT DATA ===")
df = pd.read_csv('aug_train.csv')

# Membuang fitur ID yang tidak relevan (hanya nomor urut acak)
df = df.drop(columns=['enrollee_id'])
print(f"Total baris awal dataset: {len(df)}\n")


# ==============================================================================
# STEP 2: PREPROCESSING (Duplikat, Missing Value, Outlier)
# ==============================================================================
print("=== STEP 2: PREPROCESSING ===")

# --- 2.1 Deteksi & Handling Duplikat ---
jumlah_duplikat = df.duplicated().sum()
print(f"Jumlah data duplikat terdeteksi: {jumlah_duplikat}")

if jumlah_duplikat > 0:
    df = df.drop_duplicates()
    print("STATUS: Data duplikat berhasil dihapus.\n")
else:
    print("STATUS: Aman dari duplikasi.\n")

# --- Pemisahan Tipe Data ---
kolom_kategori = df.select_dtypes(include=['object']).columns.tolist()
kolom_numerik = df.select_dtypes(exclude=['object']).columns.tolist()

# Pisahkan kolom target agar tidak ikut diproses secara matematis
if 'target' in kolom_numerik:
    kolom_numerik.remove('target')

# --- 2.2 Deteksi & Handling Missing Value ---
print("Missing value SEBELUM imputasi:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Imputasi menggunakan Modus untuk data kategorikal
for col in kolom_kategori:
    if df[col].isnull().sum() > 0:
        modus = df[col].mode()[0]
        df[col] = df[col].fillna(modus)

print("\nMissing value SETELAH imputasi:", df.isnull().sum().sum())

# --- 2.3 Auto-Deteksi & Handling Outlier ---
print("\n--- Analisis Outlier (Boxplot / IQR Capping) ---")
for col in kolom_numerik:
    kemiringan = df[col].skew()
    print(f"[{col}] Skewness: {kemiringan:.2f} (Distribusi Miring)")
    
    # Menghitung batas kuartil IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - (1.5 * IQR)
    batas_atas = Q3 + (1.5 * IQR)
    
    # Menghitung jumlah outlier sebelum di-capping
    outlier_awal = ((df[col] < batas_bawah) | (df[col] > batas_atas)).sum()
    print(f"Terdeteksi {outlier_awal} outlier -> ACTION: Ganti Batas Max & Min")
    
    # Eksekusi Capping (Membatasi nilai ekstrem tanpa menghapus baris)
    df[col] = np.where(df[col] < batas_bawah, batas_bawah, df[col])
    df[col] = np.where(df[col] > batas_atas, batas_atas, df[col])

print("\nSTATUS: Preprocessing Selesai! Data siap untuk tahap Split Data.")

# ==============================================================================
# STEP 3: DATA ENCODING & SPLIT DATA
# ==============================================================================
print("\n=== STEP 3: ENCODING & SPLIT DATA ===")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 3.1 Label Encoding (Mengubah teks menjadi angka agar bisa diproses algoritma)
le = LabelEncoder()
for col in kolom_kategori:
    df[col] = le.fit_transform(df[col])

# 3.2 Memisahkan Fitur (X) dan Target (y)
X = df.drop(columns=['target'])
y = df['target']

# 3.3 Split Data: 80% Data Training dan 20% Data Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Jumlah Data Training (80%): {len(X_train)} baris")
print(f"Jumlah Data Testing (20%): {len(X_test)} baris")

# ==============================================================================
# STEP 4: TRANSFORMASI (MIN-MAX SCALING)
# ==============================================================================
print("\n=== STEP 4: TRANSFORMASI MIN-MAX ===")
from sklearn.preprocessing import MinMaxScaler

# Membuat objek Min-Max Scaler
scaler = MinMaxScaler()

# 4.1 Menerapkan pada Data Training (Fit & Transform)
X_train[kolom_numerik] = scaler.fit_transform(X_train[kolom_numerik])

# 4.2 Menerapkan pada Data Testing (HANYA Transform, tanpa Fit)
X_test[kolom_numerik] = scaler.transform(X_test[kolom_numerik])

print("Data Training Numerik (setelah Min-Max):")
print(X_train[kolom_numerik].head(3))

# ==============================================================================
# STEP 5: SELEKSI FITUR (Chi-Square & T-test)
# ==============================================================================
print("\n=== STEP 5: SELEKSI FITUR ===")
from scipy.stats import chi2_contingency, ttest_ind

fitur_terpilih = []

# 5.1 Chi-Square untuk Kolom Kategorikal
print("--- Uji Chi-Square (Data Kategorikal) ---")
for col in kolom_kategori:
    contingency_table = pd.crosstab(X_train[col], y_train)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    if p < 0.05:
        fitur_terpilih.append(col)
        print(f"[Keep] {col} (p-value: {p:.4f}) -> Signifikan")
    else:
        print(f"[Drop] {col} (p-value: {p:.4f}) -> Tidak Signifikan")

# 5.2 T-test untuk Kolom Numerik
print("\n--- Uji T-test (Data Numerik) ---")
for col in kolom_numerik:
    # Memisahkan data numerik berdasarkan kelas target (0 dan 1) pada Data Training
    grup_0 = X_train[y_train == 0][col]
    grup_1 = X_train[y_train == 1][col]
    
    stat, p = ttest_ind(grup_0, grup_1)
    if p < 0.05:
        fitur_terpilih.append(col)
        print(f"[Keep] {col} (p-value: {p:.4f}) -> Signifikan")
    else:
        print(f"[Drop] {col} (p-value: {p:.4f}) -> Tidak Signifikan")

# 5.3 Menerapkan filter fitur ke Data Training dan Data Testing
print("\n--- Hasil Seleksi Fitur ---")
X_train_final = X_train[fitur_terpilih]
X_test_final = X_test[fitur_terpilih]
print(f"Jumlah fitur awal: {X_train.shape[1]}")
print(f"Jumlah fitur akhir: {len(fitur_terpilih)}")

# ==============================================================================
# STEP 6: EKSPOR DATA
# ==============================================================================
# menggabungkan kembali X dan y agar siap dilatih oleh algoritma Decision Tree
train_final = pd.concat([X_train_final, y_train], axis=1)
test_final = pd.concat([X_test_final, y_test], axis=1)

# Simpan ke CSV
train_final.to_csv('data_training_siap.csv', index=False)
test_final.to_csv('data_testing_siap.csv', index=False)
print("\nSTATUS: File 'data_training_siap.csv' dan 'data_testing_siap.csv' berhasil diekspor!")