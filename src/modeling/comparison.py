"""
=============================================================================
STEP 7: PERBANDINGAN VISUAL 9 KOMBINASI EKSPERIMEN
(3 Model x 3 Varian Seleksi Fitur = 9 Eksperimen)
=============================================================================
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

OUTPUT_DIR = 'assets/comparison'
METRIC_COLUMNS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

# 9 Kombinasi Spesifikasi Model & Varian Data
COMBINATIONS = [
    # --- DECISION TREE ---
    {'model_name': 'Decision Tree', 'variant': 'ORI', 'label': 'DT + ORI', 'train_csv': 'data/processed/data_training_ori.csv', 'test_csv': 'data/processed/data_testing_ori.csv', 'model_path': 'assets/decision_tree/model_dt_ori.pkl', 'color': '#15803d', 'factory': lambda: DecisionTreeClassifier(random_state=42)},
    {'model_name': 'Decision Tree', 'variant': 'CSTT', 'label': 'DT + CSTT', 'train_csv': 'data/processed/data_training_cstt.csv', 'test_csv': 'data/processed/data_testing_cstt.csv', 'model_path': 'assets/decision_tree/model_dt_cstt.pkl', 'color': '#22c55e', 'factory': lambda: DecisionTreeClassifier(random_state=42)},
    {'model_name': 'Decision Tree', 'variant': 'CSANV', 'label': 'DT + CSANV', 'train_csv': 'data/processed/data_training_csanv.csv', 'test_csv': 'data/processed/data_testing_csanv.csv', 'model_path': 'assets/decision_tree/model_dt_csanv.pkl', 'color': '#86efac', 'factory': lambda: DecisionTreeClassifier(random_state=42)},
    
    # --- SVM ---
    {'model_name': 'SVM', 'variant': 'ORI', 'label': 'SVM + ORI', 'train_csv': 'data/processed/data_training_ori.csv', 'test_csv': 'data/processed/data_testing_ori.csv', 'model_path': 'assets/svm/model_svm_ori.pkl', 'color': '#1e40af', 'factory': lambda: SVC(random_state=0, probability=True)},
    {'model_name': 'SVM', 'variant': 'CSTT', 'label': 'SVM + CSTT', 'train_csv': 'data/processed/data_training_cstt.csv', 'test_csv': 'data/processed/data_testing_cstt.csv', 'model_path': 'assets/svm/model_svm_cstt.pkl', 'color': '#3b82f6', 'factory': lambda: SVC(random_state=0, probability=True)},
    {'model_name': 'SVM', 'variant': 'CSANV', 'label': 'SVM + CSANV', 'train_csv': 'data/processed/data_training_csanv.csv', 'test_csv': 'data/processed/data_testing_csanv.csv', 'model_path': 'assets/svm/model_svm_csanv.pkl', 'color': '#93c5fd', 'factory': lambda: SVC(random_state=0, probability=True)},
    
    # --- NAIVE BAYES ---
    {'model_name': 'Naïve Bayes', 'variant': 'ORI', 'label': 'NB + ORI', 'train_csv': 'data/processed/data_training_ori.csv', 'test_csv': 'data/processed/data_testing_ori.csv', 'model_path': 'assets/naive_bayes/model_nb_ori.pkl', 'color': '#b45309', 'factory': lambda: GaussianNB()},
    {'model_name': 'Naïve Bayes', 'variant': 'CSTT', 'label': 'NB + CSTT', 'train_csv': 'data/processed/data_training_cstt.csv', 'test_csv': 'data/processed/data_testing_cstt.csv', 'model_path': 'assets/naive_bayes/model_nb_cstt.pkl', 'color': '#f59e0b', 'factory': lambda: GaussianNB()},
    {'model_name': 'Naïve Bayes', 'variant': 'CSANV', 'label': 'NB + CSANV', 'train_csv': 'data/processed/data_training_csanv.csv', 'test_csv': 'data/processed/data_testing_csanv.csv', 'model_path': 'assets/naive_bayes/model_nb_csanv.pkl', 'color': '#fde047', 'factory': lambda: GaussianNB()}
]

sns.set_theme(style='whitegrid', context='talk', font_scale=0.8)

def _positive_scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    span = scores.max() - scores.min()
    return (scores - scores.min()) / span if span else np.zeros_like(scores)

def collect_all_metrics():
    rows, cms, roc_data, timings = [], {}, {}, {}

    for spec in COMBINATIONS:
        # Load Model atau Train ulang jika pkl belum ada
        if os.path.exists(spec['model_path']):
            with open(spec['model_path'], 'rb') as f:
                model = pickle.load(f)
        else:
            df_tr = pd.read_csv(spec['train_csv'])
            model = spec['factory']()
            model.fit(df_tr.drop(columns=['target']), df_tr['target'])

        # Read Test Data
        df_te = pd.read_csv(spec['test_csv'])
        X_test, y_test = df_te.drop(columns=['target']), df_te['target']

        # Measure Train Time
        df_tr = pd.read_csv(spec['train_csv'])
        t0 = time.perf_counter()
        m_tmp = spec['factory']()
        m_tmp.fit(df_tr.drop(columns=['target']), df_tr['target'])
        timings[spec['label']] = time.perf_counter() - t0

        # Predict
        y_pred = model.predict(X_test)
        y_score = _positive_scores(model, X_test)

        acc = round(accuracy_score(y_test, y_pred) * 100, 2)
        prec = round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
        rec = round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
        f1 = round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
        auc = round(roc_auc_score(y_test, y_score) * 100, 2)

        rows.append({
            'Kombinasi': spec['label'],
            'Model': spec['model_name'],
            'Seleksi Fitur': spec['variant'],
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

        cms[spec['label']] = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_data[spec['label']] = (fpr, tpr, auc, spec['color'])

    df_res = pd.DataFrame(rows)
    return df_res, cms, roc_data, timings

# =============================================================================
# VISUALISASI 9 KOMBINASI
# =============================================================================

def plot_metrics_bar_9(df_res):
    """Bar chart perbandingan 9 kombinasi untuk 5 metrik"""
    fig, ax = plt.subplots(figsize=(16, 7))
    df_melt = df_res.melt(id_vars=['Kombinasi'], value_vars=METRIC_COLUMNS, var_name='Metrik', value_name='Skor')

    sns.barplot(data=df_melt, x='Metrik', y='Skor', hue='Kombinasi', palette='tab10', ax=ax)
    ax.set_ylim(0, 115)
    ax.set_title('Perbandingan 5 Metrik Evaluasi across 9 Kombinasi Eksperimen', fontweight='bold', pad=15)
    ax.set_ylabel('Skor (%)', fontweight='bold')
    ax.legend(title='Kombinasi Eksperimen', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'comparison_metrics_9_combos.png'), dpi=200)
    plt.close()

def plot_radar_9(df_res):
    """Radar chart 9 kombinasi"""
    labels = METRIC_COLUMNS
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'polar': True})
    
    for _, row in df_res.iterrows():
        values = row[labels].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=row['Kombinasi'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight='bold', fontsize=10)
    ax.set_title('Radar Chart Performa 9 Kombinasi Eksperimen', fontweight='bold', pad=25)
    ax.legend(bbox_to_anchor=(1.25, 1.1), loc='upper left', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'comparison_radar_9_combos.png'), dpi=200)
    plt.close()

def plot_roc_9(roc_data):
    """Kurva ROC 9 Kombinasi"""
    fig, ax = plt.subplots(figsize=(9, 7))
    for label, (fpr, tpr, auc, color) in roc_data.items():
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{label} (AUC = {auc:.1f}%)")
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_title('Kurva ROC-AUC 9 Kombinasi Eksperimen', fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'comparison_roc_9_combos.png'), dpi=200)
    plt.close()

def plot_efficiency_9(df_res, timings):
    """Scatter Trade-off Waktu vs Akurasi 9 Kombinasi"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for spec in COMBINATIONS:
        lbl = spec['label']
        t = timings[lbl]
        acc = df_res.loc[df_res['Kombinasi'] == lbl, 'Accuracy'].values[0]
        ax.scatter(t, acc, s=250, color=spec['color'], alpha=0.85, edgecolors='black')
        ax.annotate(lbl, (t, acc), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8, fontweight='bold')

    ax.set_title('Trade-off Efisiensi Waktu Training vs Akurasi (9 Kombinasi)', fontweight='bold', pad=15)
    ax.set_xlabel('Waktu Training (detik)', fontweight='bold')
    ax.set_ylabel('Akurasi Test (%)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'comparison_efficiency_9_combos.png'), dpi=200)
    plt.close()

def plot_confusion_grid_9(cms):
    """Grid 3x3 Confusion Matrix"""
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    
    for i, spec in enumerate(COMBINATIONS):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        cm = cms[spec['label']]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        ax.set_title(spec['label'], fontsize=11, fontweight='bold', color=spec['color'])
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True', fontsize=8)

    fig.suptitle('Confusion Matrix 9 Kombinasi Eksperimen (Grid 3x3)', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'comparison_confusion_9_combos.png'), dpi=200)
    plt.close()

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def run_comparison():
    print("=== STEP 7: MENJALANKAN EVALUASI & VISUALISASI 9 KOMBINASI ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_res, cms, roc_data, timings = collect_all_metrics()

    # Save CSV
    df_res.to_csv('assets/comparison_results.csv', index=False)
    
    print("\n=========================================================================")
    print("            TABEL RINGKASAN EVALUASI 9 KOMBINASI EKSPERIMEN              ")
    print("=========================================================================")
    print(df_res.to_string(index=False))
    print("=========================================================================\n")

    print("Membuat 5 grafik visualisasi 9 kombinasi ...")
    plot_metrics_bar_9(df_res)
    plot_radar_9(df_res)
    plot_roc_9(roc_data)
    plot_efficiency_9(df_res, timings)
    plot_confusion_grid_9(cms)

    print(f"STATUS: Seluruh 5 grafik 9 kombinasi berhasil disimpan di '{OUTPUT_DIR}/'.\n")
    return df_res

if __name__ == '__main__':
    run_comparison()