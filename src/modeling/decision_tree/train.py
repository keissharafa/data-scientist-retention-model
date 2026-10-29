import os
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def train_model(
    train_csv='data/processed/data_training_ori.csv',
    model_output_path='assets/decision_tree/model_dt_ori.pkl',
    variant_label='ORI'
):
    # ==============================================================================
    # STEP 5: PEMODELAN DECISION TREE
    # ==============================================================================
    print(f"=== STEP 5: DECISION TREE TRAINING ({variant_label}) ===")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"File {train_csv} tidak ditemukan! Jalankan modul preparation terlebih dahulu.")

    # Membaca data training yang sudah siap
    df_train = pd.read_csv(train_csv)
    X_train = df_train.drop(columns=['target'])
    y_train = df_train['target']

    # --- Inisialisasi & Training Model ---
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)

    print(f"STATUS: Pemodelan Decision Tree ({variant_label}) selesai dilatih.")

    # Simpan hasil model (otak AI) ke folder assets
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(decision_tree, f)

    print(f"STATUS: Model disimpan di '{model_output_path}'.\n")
    return decision_tree

if __name__ == '__main__':
    # Melatih 3 Varian Data
    train_model('data/processed/data_training_ori.csv', 'assets/decision_tree/model_dt_ori.pkl', 'ORI')
    train_model('data/processed/data_training_cstt.csv', 'assets/decision_tree/model_dt_cstt.pkl', 'CSTT')
    train_model('data/processed/data_training_csanv.csv', 'assets/decision_tree/model_dt_csanv.pkl', 'CSANV')