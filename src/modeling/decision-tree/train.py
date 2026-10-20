import os
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def train_model(train_csv='data/processed/data_training_siap.csv'):
    # ==============================================================================
    # STEP 5: PEMODELAN DECISION TREE
    # ==============================================================================
    print("=== STEP 5: DECISION TREE TRAINING ===")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"File {train_csv} tidak ditemukan! Jalankan modul preparation terlebih dahulu.")

    # Membaca data training yang sudah siap
    df_train = pd.read_csv(train_csv)
    X_train = df_train.drop(columns=['target'])
    y_train = df_train['target']

    # --- Inisialisasi & Training Model ---
    decision_tree = DecisionTreeClassifier(random_state=0)
    decision_tree.fit(X_train, y_train)

    print("STATUS: Pemodelan Decision Tree selesai dilatih.")

    # Simpan hasil model (otak AI) ke folder assets
    os.makedirs('assets', exist_ok=True)
    with open('assets/model.pkl', 'wb') as f:
        pickle.dump(decision_tree, f)

    print("STATUS: Model disimpan di 'assets/model.pkl'.\n")
    return decision_tree

if __name__ == '__main__':
    train_model()