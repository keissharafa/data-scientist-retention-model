import os
import pickle
import pandas as pd
from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(
    train_csv='data/processed/data_training_ori.csv',
    model_output_path='assets/naive_bayes/model_nb_ori.pkl',
    variant_label='ORI'
):
    # ==============================================================================
    # STEP 5: PEMODELAN NAÏVE BAYES
    # ==============================================================================
    print(f"=== STEP 5: NAÏVE BAYES TRAINING ({variant_label}) ===")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"File {train_csv} tidak ditemukan! Jalankan modul preparation terlebih dahulu.")

    # Membaca data training
    df_train = pd.read_csv(train_csv)
    X_train = df_train.drop(columns=['target'])
    y_train = df_train['target']

    # --- Inisialisasi & Training Model Naïve Bayes ---
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    print(f"STATUS: Pemodelan Naïve Bayes ({variant_label}) selesai dilatih.")

    # Menyimpan hasil model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(nb_model, f)

    print(f"STATUS: Model disimpan di '{model_output_path}'.\n")
    return nb_model

if __name__ == '__main__':
    # Melatih 3 Varian Data
    train_naive_bayes('data/processed/data_training_ori.csv', 'assets/naive_bayes/model_nb_ori.pkl', 'ORI')
    train_naive_bayes('data/processed/data_training_cstt.csv', 'assets/naive_bayes/model_nb_cstt.pkl', 'CSTT')
    train_naive_bayes('data/processed/data_training_csanv.csv', 'assets/naive_bayes/model_nb_csanv.pkl', 'CSANV')