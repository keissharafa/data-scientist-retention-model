import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_naive_bayes(
    test_csv='data/processed/data_testing_ori.csv',
    model_path='assets/naive_bayes/model_nb_ori.pkl',
    cm_output_path='assets/naive_bayes/cm_nb_ori.png',
    variant_label='ORI'
):
    # ==============================================================================
    # STEP 6: EVALUASI MODEL NAÏVE BAYES & CONFUSION MATRIX
    # ==============================================================================
    print(f"=== STEP 6: EVALUASI MODEL NAÏVE BAYES ({variant_label}) ===")

    if not os.path.exists(test_csv) or not os.path.exists(model_path):
        raise FileNotFoundError(f"File {test_csv} atau {model_path} tidak ditemukan!")

    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['target'])
    y_test = df_test['target']

    # --- Load Model Naïve Bayes ---
    with open(model_path, 'rb') as f:
        nb_model = pickle.load(f)

    # --- Prediksi Naïve Bayes ---
    print(f"instance prediksi Naïve Bayes ({variant_label}):")
    Y_pred = nb_model.predict(X_test)
    print(Y_pred)
    print("=" * 75)

    # --- Hitung Metrik Evaluasi ---
    accuracy = round(accuracy_score(y_test, Y_pred) * 100, 2)
    precision = round(precision_score(y_test, Y_pred, zero_division=0) * 100, 2)
    recall = round(recall_score(y_test, Y_pred, zero_division=0) * 100, 2)
    f1 = round(f1_score(y_test, Y_pred, zero_division=0) * 100, 2)

    print(f"\nAkurasi   : {accuracy}%")
    print(f"Presisi   : {precision}%")
    print(f"Recall    : {recall}%")
    print(f"F1-Score  : {f1}%\n")

    # --- Display Classification Report & Confusion Matrix ---
    print(f"CLASSIFICATION REPORT NAÏVE BAYES ({variant_label})".center(75, '='))
    print(classification_report(y_test, Y_pred, zero_division=0))

    cm = confusion_matrix(y_test, Y_pred)
    print("Confusion Matrix:")
    print(cm)

    # --- Visualisasi Heatmap Confusion Matrix ---
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Naïve Bayes ({variant_label})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    # Nyimpan grafik
    os.makedirs(os.path.dirname(cm_output_path), exist_ok=True)
    plt.savefig(cm_output_path)
    plt.close()
    print(f"\nSTATUS: Grafik Confusion Matrix disimpan di '{cm_output_path}'.\n")

    return {
        "Kombinasi": f"Naïve Bayes + {variant_label}",
        "Model": "Naïve Bayes",
        "Seleksi Fitur": variant_label,
        "Accuracy": f"{accuracy}%",
        "Precision": f"{precision}%",
        "Recall": f"{recall}%",
        "F1-Score": f"{f1}%"
    }

if __name__ == '__main__':
    # Mengevaluasi 3 Varian Data
    evaluate_naive_bayes('data/processed/data_testing_ori.csv', 'assets/naive_bayes/model_nb_ori.pkl', 'assets/naive_bayes/cm_nb_ori.png', 'ORI (Tanpa Seleksi)')
    evaluate_naive_bayes('data/processed/data_testing_cstt.csv', 'assets/naive_bayes/model_nb_cstt.pkl', 'assets/naive_bayes/cm_nb_cstt.png', 'CSTT (Chi-Square + T-Test)')
    evaluate_naive_bayes('data/processed/data_testing_csanv.csv', 'assets/naive_bayes/model_nb_csanv.pkl', 'assets/naive_bayes/cm_nb_csanv.png', 'CSANV (Chi-Square + ANOVA)')