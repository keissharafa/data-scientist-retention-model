import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def run_splitting(input_path='data/interim/cleaned_data.csv'):
    # ==============================================================================
    # STEP 2: DATA ENCODING & SPLIT DATA
    # ==============================================================================
    print("=== STEP 2: ENCODING & SPLIT DATA ===")
    df = pd.read_csv(input_path)

    kolom_kategori = df.select_dtypes(include=['object']).columns.tolist()

    # --- 2.1 Label Encoding ---
    le = LabelEncoder()
    encoders = {}
    for col in kolom_kategori:
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Simpan encoder ke assets untuk dipanggil saat inference
    os.makedirs('assets', exist_ok=True)
    with open('assets/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    # --- 2.2 Memisahkan Fitur (X) dan Target (y) ---
    X = df.drop(columns=['target'])
    y = df['target']

    # --- 2.3 Split Data: 80% Data Training dan 20% Data Testing ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print(f"Jumlah Data Training (80%): {len(X_train)} baris")
    print(f"Jumlah Data Testing (20%): {len(X_test)} baris")
    print("STATUS: Encoding dan Splitting Data selesai.\n")
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    run_splitting()