import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def run_preprocessing(input_path='data/raw/aug_train.csv', output_path='data/interim/cleaned_data.csv'):
    # ==============================================================================
    # STEP 1: PREPROCESSING (Duplikat, Missing Value, Outlier)
    # ==============================================================================
    print("=== STEP 1: PREPROCESSING DATA ===")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} tidak ditemukan! Pastikan file aug_train.csv ada di folder data/raw/")

    df = pd.read_csv(input_path)

    # Membuang fitur ID yang tidak relevan (hanya nomor urut acak)
    if 'enrollee_id' in df.columns:
        df = df.drop(columns=['enrollee_id'])
    print(f"Total baris awal dataset: {len(df)}\n")

    # --- 1.1 Deteksi & Handling Duplikat ---
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

    if 'target' in kolom_numerik:
        kolom_numerik.remove('target')

    # --- 1.2 Deteksi & Handling Missing Value ---
    print("Missing value SEBELUM imputasi:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Imputasi menggunakan Modus untuk data kategorikal
    for col in kolom_kategori:
        if df[col].isnull().sum() > 0:
            modus = df[col].mode()[0]
            df[col] = df[col].fillna(modus)

    # Imputasi menggunakan Median untuk data numerik
    for col in kolom_numerik:
        if df[col].isnull().sum() > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    print("\nMissing value SETELAH imputasi:", df.isnull().sum().sum())

    # --- 1.3 Auto-Deteksi & Handling Outlier ---
    print("\n--- Analisis Outlier (Boxplot / IQR Capping) ---")
    for col in kolom_numerik:
        kemiringan = df[col].skew()
        print(f"[{col}] Skewness: {kemiringan:.2f} (Distribusi Miring)")
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        batas_bawah = Q1 - (1.5 * IQR)
        batas_atas = Q3 + (1.5 * IQR)
        
        outlier_awal = ((df[col] < batas_bawah) | (df[col] > batas_atas)).sum()
        print(f"Terdeteksi {outlier_awal} outlier -> ACTION: Ganti Batas Max & Min")
        
        df[col] = np.where(df[col] < batas_bawah, batas_bawah, df[col])
        df[col] = np.where(df[col] > batas_atas, batas_atas, df[col])

    # Ekspor data bersih ke folder interim
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSTATUS: Preprocessing Selesai! Data disimpan ke '{output_path}'.\n")
    return df

if __name__ == '__main__':
    run_preprocessing()