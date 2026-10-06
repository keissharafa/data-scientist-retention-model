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