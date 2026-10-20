import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_model_svm(test_csv='data/processed/data_testing_siap.csv', model_path='assets/model_svm.pkl'):
    # ==============================================================================
    # STEP 6: EVALUASI MODEL SVM & CONFUSION MATRIX
    # ==============================================================================
    print("=== STEP 6: EVALUASI MODEL SVM ===")

    if not os.path.exists(test_csv) or not os.path.exists(model_path):
        raise FileNotFoundError("File data_testing_siap.csv atau model_svm.pkl tidak ditemukan!")

    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['target'])
    y_test = df_test['target']

    # --- Load Model ---
    with open(model_path, 'rb') as f:
        svm_model = pickle.load(f)

    # --- Prediksi SVM ---
    print("instance prediksi SVM:")
    Y_pred = svm_model.predict(X_test)
    print(Y_pred)
    print("=" * 75)

    # --- Hitung Prediksi Akurasi ---
    accuracy = round(accuracy_score(y_test, Y_pred) * 100, 2)
    precision = round(precision_score(y_test, Y_pred, average='weighted') * 100, 2)
    recall = round(recall_score(y_test, Y_pred, average='weighted') * 100, 2)
    f1 = round(f1_score(y_test, Y_pred, average='weighted') * 100, 2)

    print(f"\nAkurasi   : {accuracy}%")
    print(f"Presisi   : {precision}%")
    print(f"Recall    : {recall}%")
    print(f"F1-Score  : {f1}%\n")

    # --- Display Classification Report & Confusion Matrix ---
    print("CLASSIFICATION REPORT SVM".center(75, '='))
    print(classification_report(y_test, Y_pred))

    cm = confusion_matrix(y_test, Y_pred)
    print("Confusion Matrix:")
    print(cm)

    # --- Visualisasi Heatmap Confusion Matrix ---
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - SVM')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/confusion_matrix_svm.png')
    print("\nSTATUS: Grafik Confusion Matrix disimpan di 'assets/confusion_matrix_svm.png'.")
    plt.show()

if __name__ == '__main__':
    evaluate_model_svm()