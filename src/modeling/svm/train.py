import os
import pickle
import pandas as pd
from sklearn.svm import SVC

def train_model_svm(
    train_csv='data/processed/data_training_ori.csv',
    model_output_path='assets/svm/model_svm_ori.pkl',
    variant_label='ORI'
):
    # ==============================================================================
    # STEP 5: PEMODELAN SUPPORT VECTOR MACHINE (SVM)
    # ==============================================================================
    print(f"=== STEP 5: SVM TRAINING ({variant_label}) ===")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"File {train_csv} tidak ditemukan! Jalankan modul preparation terlebih dahulu.")

    # Membaca data training yang sudah siap
    df_train = pd.read_csv(train_csv)
    X_train = df_train.drop(columns=['target'])
    y_train = df_train['target']

    # --- Inisialisasi & Training Model ---
    svm_model = SVC(random_state=0) 
    svm_model.fit(X_train, y_train)

    print(f"STATUS: Pemodelan SVM ({variant_label}) selesai dilatih.")

    # Simpan hasil model ke folder assets
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(svm_model, f)

    print(f"STATUS: Model disimpan di '{model_output_path}'.\n")
    return svm_model

if __name__ == '__main__':
    # Melatih 3 Varian Data
    train_model_svm('data/processed/data_training_ori.csv', 'assets/svm/model_svm_ori.pkl', 'ORI')
    train_model_svm('data/processed/data_training_cstt.csv', 'assets/svm/model_svm_cstt.pkl', 'CSTT')
    train_model_svm('data/processed/data_training_csanv.csv', 'assets/svm/model_svm_csanv.pkl', 'CSANV')