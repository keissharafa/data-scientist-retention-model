"""
=============================================================================
STEP 7: PERBANDINGAN VISUAL MODEL (DECISION TREE vs SVM vs NAÏVE BAYES)
=============================================================================
Menghasilkan 5 visualisasi pembanding:
  1. comparison_metrics.png      -> Grouped bar chart 5 metrik
  2. comparison_confusion.png    -> 3 confusion matrix berdampingan
  3. comparison_roc.png          -> Kurva ROC bertumpuk
  4. comparison_radar.png        -> Radar chart performa multidimensi
  5. comparison_efficiency.png   -> Trade-off akurasi vs waktu training

Semua hasil disimpan ke folder 'assets/comparison/'.
Catatan: hanya untuk visualisasi, TIDAK mengubah model .pkl yang sudah ada.
=============================================================================
"""

import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# =============================================================================
# KONFIGURASI
# =============================================================================
TEST_CSV = 'data/processed/data_testing_siap.csv'
TRAIN_CSV = 'data/processed/data_training_siap.csv'
OUTPUT_DIR = 'assets/comparison'

MODELS = [
    {
        'name': 'Decision Tree',
        'short': 'DT',
        'path': 'assets/model.pkl',
        'factory': lambda: DecisionTreeClassifier(random_state=0),
        'color': '#22c55e',
        'cmap': 'Greens',
    },
    {
        'name': 'SVM',
        'short': 'SVM',
        'path': 'assets/model_svm.pkl',
        'factory': lambda: SVC(random_state=0),
        'color': '#6366f1',
        'cmap': 'Blues',
    },
    {
        'name': 'Naïve Bayes',
        'short': 'NB',
        'path': 'assets/naive_bayes_model.pkl',
        'factory': lambda: GaussianNB(),
        'color': '#f59e0b',
        'cmap': 'Oranges',
    },
]

METRIC_COLUMNS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

sns.set_theme(style='whitegrid', context='talk', font_scale=0.85)
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.edgecolor': '#94a3b8',
    'axes.labelcolor': '#1e293b',
    'text.color': '#1e293b',
    'xtick.color': '#334155',
    'ytick.color': '#334155',
})


# =============================================================================
# 1. PENGUMPULAN METRIK
# =============================================================================
def _positive_scores(model, X):
    """Skor kelas positif: predict_proba bila ada, jika tidak decision_function."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    # normalisasi ke [0, 1] agar sebanding dengan probabilitas pada kurva ROC
    span = scores.max() - scores.min()
    return (scores - scores.min()) / span if span else np.zeros_like(scores)


def _load_model(spec):
    """Muat model .pkl; fallback latih ulang bila file tidak tersedia."""
    if os.path.exists(spec['path']):
        with open(spec['path'], 'rb') as f:
            return pickle.load(f)
    print(f"[WARN] {spec['path']} tidak ditemukan → melatih ulang {spec['name']}.")
    return _train_and_time(spec)[0]


def _train_and_time(spec, X_train=None, y_train=None):
    """Latih model dari nol dan ukur waktu training (detik)."""
    if X_train is None:
        df = pd.read_csv(TRAIN_CSV)
        X_train, y_train = df.drop(columns=['target']), df['target']
    start = time.perf_counter()
    model = spec['factory']()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return model, elapsed


def collect_metrics(measure_time=True):
    """Kembalikan (DataFrame metrik, dict confusion matrix, dict ROC, dict waktu)."""
    df_train = pd.read_csv(TRAIN_CSV)
    X_train, y_train = df_train.drop(columns=['target']), df_train['target']

    df_test = pd.read_csv(TEST_CSV)
    X_test, y_test = df_test.drop(columns=['target']), df_test['target']

    rows, cms, roc_data, timings = [], {}, {}, {}

    for spec in MODELS:
        model = _load_model(spec)
        y_pred = model.predict(X_test)
        y_score = _positive_scores(model, X_test)

        rows.append({
            'Model': spec['name'],
            'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
            'Recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
            'F1-Score': round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
            'ROC-AUC': round(roc_auc_score(y_test, y_score) * 100, 2),
        })

        cms[spec['name']] = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_data[spec['name']] = (fpr, tpr, roc_auc_score(y_test, y_score))

        if measure_time:
            print(f"  · mengukur waktu training {spec['name']} ...")
            timings[spec['name']] = round(_train_and_time(spec, X_train, y_train)[1], 2)

    return pd.DataFrame(rows).set_index('Model'), cms, roc_data, timings


# =============================================================================
# 2. GROUPED BAR CHART — 5 METRIK
# =============================================================================
def plot_metric_comparison(df, save=True):
    fig, ax = plt.subplots(figsize=(13, 6.5))

    x = np.arange(len(METRIC_COLUMNS))
    width = 0.26

    for i, spec in enumerate(MODELS):
        values = df.loc[spec['name'], METRIC_COLUMNS].values
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, values, width,
            label=spec['name'], color=spec['color'],
            edgecolor='white', linewidth=1.2, alpha=0.92,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + 0.9,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#0f172a',
            )

    # Sorot model terbaik per metrik
    short_of = {spec['name']: spec['short'] for spec in MODELS}
    for j, metric in enumerate(METRIC_COLUMNS):
        best_model = df[metric].idxmax()
        ax.text(
            j, 100.5, f'* {short_of[best_model]}',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#b45309',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_COLUMNS, fontweight='bold')
    ax.set_ylabel('Skor (%)', fontweight='bold')
    ax.set_ylim(0, 108)
    ax.set_title(
        'Perbandingan Performa Model Klasifikasi\n'
        'Decision Tree vs Support Vector Machine vs Naïve Bayes  ',
        fontsize=15, fontweight='bold', pad=18,
    )
    ax.text(
        0.99, 1.045, '* = model terbaik pada metrik tersebut',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8.5, style='italic', color='#b45309',
    )
    ax.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    if save:
        _save(fig, 'comparison_metrics.png')
    return fig


# =============================================================================
# 3. CONFUSION MATRIX BERSANDINGAN
# =============================================================================
def plot_confusion_grid(cms, save=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for ax, spec in zip(axes, MODELS):
        cm = cms[spec['name']]
        sns.heatmap(
            cm, annot=True, fmt=',d', cmap=spec['cmap'], cbar=False,
            ax=ax, linewidths=1.5, linecolor='white',
            annot_kws={'fontsize': 15, 'fontweight': 'bold'},
        )
        ax.set_title(spec['name'], fontsize=13, fontweight='bold', color=spec['color'], pad=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xticklabels(['Stay (0)', 'Leave (1)'], fontsize=9)
        ax.set_yticklabels(['Stay (0)', 'Leave (1)'], fontsize=9, rotation=0)

        # Anotasi turunan metrik di bawah tiap sel
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / cm.sum() * 100
        ax.text(
            0.5, -0.30,
            f'Akurasi {acc:.2f}%  |  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, color='#475569',
        )

    fig.suptitle(
        'Confusion Matrix per Model (Test Set)',
        fontsize=15, fontweight='bold', y=1.06,
    )
    fig.tight_layout()
    if save:
        _save(fig, 'comparison_confusion.png')
    return fig


# =============================================================================
# 4. KURVA ROC
# =============================================================================
def plot_roc(roc_data, save=True):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for spec in MODELS:
        fpr, tpr, auc = roc_data[spec['name']]
        ax.plot(
            fpr, tpr, color=spec['color'], lw=2.6,
            label=f"{spec['name']} (AUC = {auc*100:.2f}%)",
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random Classifier (50%)')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('Kurva ROC-AUC Perbandingan Model', fontsize=14, fontweight='bold', pad=14)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

    if save:
        _save(fig, 'comparison_roc.png')
    return fig


# =============================================================================
# 5. RADAR CHART
# =============================================================================
def plot_radar(df, save=True):
    labels = METRIC_COLUMNS
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.2), subplot_kw={'polar': True})

    lo = np.floor(df[labels].values.min() / 5) * 5 - 5
    hi = 100

    for spec in MODELS:
        values = df.loc[spec['name'], labels].values.tolist()
        values += values[:1]
        ax.plot(angles, values, color=spec['color'], lw=2.4, label=spec['name'])
        ax.fill(angles, values, color=spec['color'], alpha=0.14)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(lo, hi + 1, 5))
    ax.set_yticklabels([f'{v:.0f}%' for v in np.arange(lo, hi + 1, 5)], fontsize=8, color='#64748b')
    ax.set_title(
        'Radar Performa Multidimensi\n(semakin lebar area, semakin baik)',
        fontsize=14, fontweight='bold', pad=28,
    )
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.12), frameon=True,
              facecolor='white', edgecolor='#cbd5e1')
    ax.grid(color='#cbd5e1', alpha=0.6)

    if save:
        _save(fig, 'comparison_radar.png')
    return fig


# =============================================================================
# 6. TRADE-OFF AKURASI vs WAKTU TRAINING
# =============================================================================
def plot_efficiency(df, timings, save=True):
    if not timings:
        return None

    fig, ax = plt.subplots(figsize=(9.5, 6))

    for spec in MODELS:
        t = timings.get(spec['name'], 0)
        acc = df.loc[spec['name'], 'Accuracy']
        ax.scatter(t, acc, s=760, color=spec['color'], alpha=0.85,
                   edgecolor='white', linewidth=2, zorder=3)
        ax.annotate(
            f"{spec['name']}\n{t:.2f}s · {acc:.2f}%",
            (t, acc), textcoords='offset points', xytext=(0, -46),
            ha='center', fontsize=10, fontweight='bold', color=spec['color'],
        )

    ax.set_xlabel('Waktu Training (detik) — semakin kecil semakin efisien', fontweight='bold')
    ax.set_ylabel('Akurasi Test (%)', fontweight='bold')
    ax.set_title('Trade-off Efisiensi Komputasi vs Akurasi', fontsize=14, fontweight='bold', pad=14)
    ax.set_xlim(left=0)
    ax.margins(y=0.28, x=0.22)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(linestyle='--', alpha=0.4)

    if save:
        _save(fig, 'comparison_efficiency.png')
    return fig


# =============================================================================
# HELPER
# =============================================================================
def _save(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    print(f"  ✔ disimpan: {path}")
    plt.close(fig)


def print_summary(df, timings):
    print('\n' + '=' * 78)
    print('RINGKASAN PERBANDINGAN MODEL'.center(78))
    print('=' * 78)
    print(df.to_string())

    print('\n--- Model Terbaik per Metrik ---')
    for metric in METRIC_COLUMNS:
        best = df[metric].idxmax()
        print(f"  {metric:<10}: {best:<14} ({df.loc[best, metric]:.2f}%)")

    if timings:
        print('\n--- Waktu Training ---')
        for name, t in timings.items():
            print(f"  {name:<14}: {t:.2f} detik")

    overall = df.mean(axis=1).idxmax()
    print(f"\nModel dengan performa rata-rata terbaik: {overall} "
          f"(rata-rata {df.loc[overall].mean():.2f}%)")
    print('=' * 78 + '\n')


# =============================================================================
# MAIN
# =============================================================================
def run_comparison():
    print('=== STEP 7: PERBANDINGAN VISUAL MODEL ===')
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"File {TEST_CSV} tidak ditemukan! Jalankan pipeline preparation terlebih dahulu."
        )

    df, cms, roc_data, timings = collect_metrics(measure_time=True)
    print_summary(df, timings)

    print('\nMembuat visualisasi ...')
    plot_metric_comparison(df)
    plot_confusion_grid(cms)
    plot_roc(roc_data)
    plot_radar(df)
    plot_efficiency(df, timings)

    print(f"\nSTATUS: 5 grafik perbandingan tersimpan di '{OUTPUT_DIR}/'.")
    return df


if __name__ == '__main__':
    run_comparison()
