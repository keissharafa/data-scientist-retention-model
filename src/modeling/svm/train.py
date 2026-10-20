import os
import pickle
import pandas as pd
from sklearn.svm import SVC

def train_model_svm(train_csv='data/processed/data_training_siap.csv'):
    # ==============================================================================
    # STEP 5: PEMODELAN SUPPORT VECTOR MACHINE (SVM)
    # ==============================================================================
    print("=== STEP 5: SVM TRAINING ===")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"File {train_csv} tidak ditemukan! Jalankan modul preparation terlebih dahulu.")

    # Membaca data training yang sudah siap
    df_train = pd.read_csv(train_csv)
    X_train = df_train.drop(columns=['target'])
    y_train = df_train['target']

    # --- Inisialisasi & Training Model ---
    # Menggunakan SVC() sesuai instruksi di PDF Pertemuan 8
    svm_model = SVC(random_state=0) 
    svm_model.fit(X_train, y_train)

    print("STATUS: Pemodelan SVM selesai dilatih.")

    # Simpan hasil model (otak AI) ke folder assets dengan nama berbeda
    os.makedirs('assets', exist_ok=True)
    with open('assets/model_svm.pkl', 'wb') as f:
        pickle.dump(svm_model, f)

    print("STATUS: Model disimpan di 'assets/model_svm.pkl'.\n")
    return svm_model

if __name__ == '__main__':
    train_model_svm()