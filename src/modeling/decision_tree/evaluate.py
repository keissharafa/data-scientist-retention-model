import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_model(
    test_csv='data/processed/data_testing_ori.csv',
    model_path='assets/decision_tree/model_dt_ori.pkl',
    cm_output_path='assets/decision_tree/cm_dt_ori.png',
    variant_label='ORI'
):
    # ==============================================================================
    # STEP 6: EVALUASI MODEL & CONFUSION MATRIX
    # ==============================================================================
    print(f"=== STEP 6: EVALUASI MODEL DECISION TREE ({variant_label}) ===")

    if not os.path.exists(test_csv) or not os.path.exists(model_path):
        raise FileNotFoundError(f"File {test_csv} atau {model_path} tidak ditemukan!")

    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=['target'])
    y_test = df_test['target']

    # --- Load Model ---
    with open(model_path, 'rb') as f:
        decision_tree = pickle.load(f)

    # --- Prediksi Decision Tree ---
    print(f"Instance prediksi decision tree ({variant_label}):")
    Y_pred = decision_tree.predict(X_test)
    print(Y_pred)
    print("=" * 75)

    # --- Hitung Prediksi Akurasi (Sesuai Modul 7) ---
    accuracy = round(accuracy_score(y_test, Y_pred) * 100, 2)
    precision = round(precision_score(y_test, Y_pred, zero_division=0) * 100, 2)
    recall = round(recall_score(y_test, Y_pred, zero_division=0) * 100, 2)
    f1 = round(f1_score(y_test, Y_pred, zero_division=0) * 100, 2)

    print(f"\nAkurasi   : {accuracy}%")
    print(f"Presisi   : {precision}%")
    print(f"Recall    : {recall}%")
    print(f"F1-Score  : {f1}%\n")

    # --- Display Classification Report & Confusion Matrix (Sesuai Modul 7) ---
    print(f"CLASSIFICATION REPORT DECISION TREE ({variant_label})".center(75, '='))
    print(classification_report(y_test, Y_pred, zero_division=0))

    cm = confusion_matrix(y_test, Y_pred)
    print("Confusion Matrix:")
    print(cm)

    # --- Visualisasi Heatmap Confusion Matrix ---
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Decision Tree ({variant_label})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    os.makedirs(os.path.dirname(cm_output_path), exist_ok=True)
    plt.savefig(cm_output_path)
    plt.close()
    print(f"\nSTATUS: Grafik Confusion Matrix disimpan di '{cm_output_path}'.\n")

    return {
        "Kombinasi": f"Decision Tree + {variant_label}",
        "Model": "Decision Tree",
        "Seleksi Fitur": variant_label,
        "Accuracy": f"{accuracy}%",
        "Precision": f"{precision}%",
        "Recall": f"{recall}%",
        "F1-Score": f"{f1}%"
    }

if __name__ == '__main__':
    # Mengevaluasi 3 Varian Data
    evaluate_model('data/processed/data_testing_ori.csv', 'assets/decision_tree/model_dt_ori.pkl', 'assets/decision_tree/cm_dt_ori.png', 'ORI (Tanpa Seleksi)')
    evaluate_model('data/processed/data_testing_cstt.csv', 'assets/decision_tree/model_dt_cstt.pkl', 'assets/decision_tree/cm_dt_cstt.png', 'CSTT (Chi-Square + T-Test)')
    evaluate_model('data/processed/data_testing_csanv.csv', 'assets/decision_tree/model_dt_csanv.pkl', 'assets/decision_tree/cm_dt_csanv.png', 'CSANV (Chi-Square + ANOVA)')