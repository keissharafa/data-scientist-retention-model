import os
import json
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

def run_feature_selection(X_train, X_test, y_train, y_test):
    # ==============================================================================
    # STEP 4: SELEKSI FITUR (Chi-Square & T-test)
    # ==============================================================================
    print("=== STEP 4: SELEKSI FITUR ===")

    kolom_kategori = [col for col in X_train.columns if X_train[col].nunique() < 15]
    kolom_numerik = [col for col in X_train.columns if col not in kolom_kategori]

    fitur_terpilih = []

    # --- 4.1 Chi-Square untuk Data Kategorikal ---
    print("--- Uji Chi-Square (Data Kategorikal) ---")
    for col in kolom_kategori:
        contingency_table = pd.crosstab(X_train[col], y_train)
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        if p < 0.05:
            fitur_terpilih.append(col)
            print(f"[Keep] {col} (p-value: {p:.4f}) -> Signifikan")
        else:
            print(f"[Drop] {col} (p-value: {p:.4f}) -> Tidak Signifikan")

    # --- 4.2 T-test untuk Data Numerik ---
    print("\n--- Uji T-test (Data Numerik) ---")
    for col in kolom_numerik:
        grup_0 = X_train[y_train == 0][col]
        grup_1 = X_train[y_train == 1][col]
        stat, p = ttest_ind(grup_0, grup_1)
        if p < 0.05:
            fitur_terpilih.append(col)
            print(f"[Keep] {col} (p-value: {p:.4f}) -> Signifikan")
        else:
            print(f"[Drop] {col} (p-value: {p:.4f}) -> Tidak Signifikan")

    # --- 4.3 Menerapkan filter fitur ke Data Training dan Data Testing ---
    print("\n--- Hasil Seleksi Fitur ---")
    X_train_final = X_train[fitur_terpilih]
    X_test_final = X_test[fitur_terpilih]
    print(f"Jumlah fitur awal: {X_train.shape[1]}")
    print(f"Jumlah fitur akhir: {len(fitur_terpilih)}")

    # Simpan daftar nama fitur terpilih
    os.makedirs('assets', exist_ok=True)
    with open('assets/selected_features.json', 'w') as f:
        json.dump(fitur_terpilih, f, indent=4)

    # --- 4.4 Ekspor Data Siap Pakai ---
    train_final = pd.concat([X_train_final, y_train], axis=1)
    test_final = pd.concat([X_test_final, y_test], axis=1)

    os.makedirs('data/processed', exist_ok=True)
    train_final.to_csv('data/processed/data_training_siap.csv', index=False)
    test_final.to_csv('data/processed/data_testing_siap.csv', index=False)
    print("\nSTATUS: File 'data_training_siap.csv' dan 'data_testing_siap.csv' berhasil diekspor!\n")

    return train_final, test_final

if __name__ == '__main__':
    pass