import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_model(test_csv='data/processed/data_testing_siap.csv', model_path='assets/model.pkl'):
    # ==============================================================================
    # STEP 6: EVALUASI MODEL & CONFUSION MATRIX
    # ==============================================================================
    print("=== STEP 6: EVALUASI MODEL ===")

    if not os.path.exists(test_csv) or not os.path.exists(model_path):
        raise FileNotFoundError("File data_testing_siap.csv atau model.pkl tidak ditemukan!")

    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['target'])
    y_test = df_test['target']

    # --- Load Model ---
    with open(model_path, 'rb') as f:
        decision_tree = pickle.load(f)

    # --- Prediksi Decision Tree ---
    print("instance prediksi decision tree:")
    Y_pred = decision_tree.predict(X_test)
    print(Y_pred)
    print("=" * 75)

    # --- Hitung Prediksi Akurasi (Sesuai Modul 7) ---
    accuracy = round(accuracy_score(y_test, Y_pred) * 100, 2)
    precision = round(precision_score(y_test, Y_pred, average='weighted') * 100, 2)
    recall = round(recall_score(y_test, Y_pred, average='weighted') * 100, 2)
    f1 = round(f1_score(y_test, Y_pred, average='weighted') * 100, 2)

    print(f"\nAkurasi   : {accuracy}%")
    print(f"Presisi   : {precision}%")
    print(f"Recall    : {recall}%")
    print(f"F1-Score  : {f1}%\n")

    # --- Display Classification Report & Confusion Matrix (Sesuai Modul 7) ---
    print("CLASSIFICATION REPORT DECISION TREE".center(75, '='))
    print(classification_report(y_test, Y_pred))

    cm = confusion_matrix(y_test, Y_pred)
    print("Confusion Matrix:")
    print(cm)

    # --- Visualisasi Heatmap Confusion Matrix ---
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Decision Tree')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/confusion_matrix.png')
    print("\nSTATUS: Grafik Confusion Matrix disimpan di 'assets/confusion_matrix.png'.")
    plt.show()

if __name__ == '__main__':
    evaluate_model()