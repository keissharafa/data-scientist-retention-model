import os
import pickle
import pandas as pd
from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(train_csv='data/processed/data_training_siap.csv'):
    print("=== STEP 5 (NB): NAÏVE BAYES TRAINING ===")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"File {train_csv} tidak ditemukan! Pastikan data siap pakai sudah ada.")

    # Membaca data training
    df_train = pd.read_csv(train_csv)
    X_train = df_train.drop(columns=['target'])
    y_train = df_train['target']

    # --- Inisialisasi & Training Model Naïve Bayes ---
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    print("STATUS: Pemodelan Naïve Bayes selesai dilatih.")

    # menyimpan hasil model 
    os.makedirs('assets', exist_ok=True)
    with open('assets/naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)

    print("STATUS: Model disimpan di 'assets/naive_bayes_model.pkl'.\n")
    return nb_model

if __name__ == '__main__':
    train_naive_bayes()