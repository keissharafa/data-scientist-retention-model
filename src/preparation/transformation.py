import os
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def run_transformation(X_train, X_test):
    # ==============================================================================
    # STEP 3: TRANSFORMASI (MIN-MAX SCALING)
    # ==============================================================================
    print("=== STEP 3: TRANSFORMASI MIN-MAX ===")
    
    kolom_numerik = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    scaler = MinMaxScaler()

    # Copy dataframe agar tidak mengubah struktur asli secara tidak sengaja
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # --- 3.1 Menerapkan pada Data Training (Fit & Transform) ---
    X_train_scaled[kolom_numerik] = scaler.fit_transform(X_train[kolom_numerik])

    # --- 3.2 Menerapkan pada Data Testing (HANYA Transform, tanpa Fit) ---
    X_test_scaled[kolom_numerik] = scaler.transform(X_test[kolom_numerik])

    # Simpan objek scaler ke folder assets
    os.makedirs('assets', exist_ok=True)
    with open('assets/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Data Training Numerik (setelah Min-Max):")
    print(X_train_scaled[kolom_numerik].head(3))
    print("STATUS: Transformasi Min-Max Selesai.\n")

    return X_train_scaled, X_test_scaled

if __name__ == '__main__':
    pass