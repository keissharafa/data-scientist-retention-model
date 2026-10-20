import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_naive_bayes(test_csv='data/processed/data_testing_siap.csv', model_path='assets/naive_bayes_model.pkl'):
    print("=== STEP 6 (NB): EVALUASI NAÏVE BAYES ===")

    if not os.path.exists(test_csv) or not os.path.exists(model_path):
        raise FileNotFoundError("File data testing atau naive_bayes_model.pkl tidak ditemukan!")

    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['target'])
    y_test = df_test['target']

    # --- Load Model Naïve Bayes ---
    with open(model_path, 'rb') as f:
        nb_model = pickle.load(f)

    # --- Prediksi Naïve Bayes ---
    Y_pred = nb_model.predict(X_test)

    # --- Hitung Metrik Evaluasi ---
    accuracy = round(accuracy_score(y_test, Y_pred) * 100, 2)
    precision = round(precision_score(y_test, Y_pred, average='weighted') * 100, 2)
    recall = round(recall_score(y_test, Y_pred, average='weighted') * 100, 2)
    f1 = round(f1_score(y_test, Y_pred, average='weighted') * 100, 2)

    print(f"\nAkurasi   : {accuracy}%")
    print(f"Presisi   : {precision}%")
    print(f"Recall    : {recall}%")
    print(f"F1-Score  : {f1}%\n")

    print("CLASSIFICATION REPORT NAÏVE BAYES".center(75, '='))
    print(classification_report(y_test, Y_pred))

    # --- Visualisasi Heatmap Confusion Matrix ---
    cm = confusion_matrix(y_test, Y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Naïve Bayes')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    # nyimpan grafik
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/confusion_matrix_nb.png')
    print("\nSTATUS: Grafik Confusion Matrix disimpan di 'assets/confusion_matrix_nb.png'.")
    plt.show()

if __name__ == '__main__':
    evaluate_naive_bayes()