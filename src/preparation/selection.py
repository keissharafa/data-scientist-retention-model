import os
import json
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

def run_feature_selection(X_train, X_test, y_train, y_test):
    # ==============================================================================
    # STEP 4: SELEKSI FITUR (MENGHASILKAN DATA T-TEST & ANOVA)
    # ==============================================================================
    print("=== STEP 4: SELEKSI FITUR ===")

    kolom_kategori = [col for col in X_train.columns if X_train[col].nunique() < 15]
    kolom_numerik = [col for col in X_train.columns if col not in kolom_kategori]

    fitur_cat_terpilih = []

    # --- 4.1 Chi-Square untuk Data Kategorikal ---
    print("--- Uji Chi-Square (Data Kategorikal) ---")
    for col in kolom_kategori:
        contingency_table = pd.crosstab(X_train[col], y_train)
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        if p < 0.05:
            fitur_cat_terpilih.append(col)
            print(f"[Keep] {col} (p-value: {p:.4f}) -> Signifikan")

    # --- 4.2 T-test untuk Data Numerik ---
    print("\n--- Uji T-test (Data Numerik) ---")
    fitur_num_ttest = []
    for col in kolom_numerik:
        grup_0 = X_train[y_train == 0][col]
        grup_1 = X_train[y_train == 1][col]
        stat, p = ttest_ind(grup_0, grup_1)
        if p < 0.05:
            fitur_num_ttest.append(col)
            print(f"[Keep] {col} (p-value: {p:.4f}) -> Signifikan")

    # --- 4.3 ANOVA untuk Data Numerik ---
    print("\n--- Uji ANOVA (Data Numerik) ---")
    fitur_num_anova = []
    for col in kolom_numerik:
        grup_0 = X_train[y_train == 0][col]
        grup_1 = X_train[y_train == 1][col]
        stat, p = f_oneway(grup_0, grup_1)
        if p < 0.05:
            fitur_num_anova.append(col)
            print(f"[Keep] {col} (p-value: {p:.4f}) -> Signifikan")

    # --- 4.4 Gabung Fitur dan Ekspor ---
    os.makedirs('data/processed', exist_ok=True)
    
    # Dataset 1: CS + T-Test
    fitur_cstt = fitur_cat_terpilih + fitur_num_ttest
    train_cstt = pd.concat([X_train[fitur_cstt], y_train], axis=1)
    test_cstt = pd.concat([X_test[fitur_cstt], y_test], axis=1)
    train_cstt.to_csv('data/processed/data_training_cstt.csv', index=False)
    test_cstt.to_csv('data/processed/data_testing_cstt.csv', index=False)

    # Dataset 2: CS + ANOVA
    fitur_csanv = fitur_cat_terpilih + fitur_num_anova
    train_csanv = pd.concat([X_train[fitur_csanv], y_train], axis=1)
    test_csanv = pd.concat([X_test[fitur_csanv], y_test], axis=1)
    train_csanv.to_csv('data/processed/data_training_csanv.csv', index=False)
    test_csanv.to_csv('data/processed/data_testing_csanv.csv', index=False)

    # Simpan log fitur untuk laporan
    os.makedirs('assets', exist_ok=True)
    with open('assets/selected_features_cstt.json', 'w') as f:
        json.dump(fitur_cstt, f, indent=4)
    with open('assets/selected_features_csanv.json', 'w') as f:
        json.dump(fitur_csanv, f, indent=4)

    print("\nSTATUS: File dataset CSTT (T-test) dan CSANV (ANOVA) berhasil diekspor!\n")

    # Kembalikan salah satu dataset agar tidak error di main pipeline
    return train_csanv, test_csanv

if __name__ == '__main__':
    pass