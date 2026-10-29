import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from preparation.preprocessing import run_preprocessing
from preparation.splitter import run_splitting
from preparation.selection_cstt import run_selection_ttest
from preparation.selection_csanv import run_selection_anova

from modeling.decision_tree.train import train_model as train_dt
from modeling.svm.train import train_model_svm as train_svm
from modeling.naive_bayes.train import train_naive_bayes as train_nb

from modeling.comparison import run_comparison

def main():
    print("==========================================================")
    print("      PIPELINE KLASIFIKASI 9 KOMBINASI EKSPERIMEN         ")
    print("==========================================================\n")

    # 1. Preparation
    run_preprocessing()
    X_tr, X_te, y_tr, y_te = run_splitting()
    run_selection_ttest(X_tr, X_te, y_tr, y_te)
    run_selection_anova(X_tr, X_te, y_tr, y_te)

    # 2. Training Decision Tree (3 Varian)
    train_dt('data/processed/data_training_ori.csv', 'assets/decision_tree/model_dt_ori.pkl', 'ORI')
    train_dt('data/processed/data_training_cstt.csv', 'assets/decision_tree/model_dt_cstt.pkl', 'CSTT')
    train_dt('data/processed/data_training_csanv.csv', 'assets/decision_tree/model_dt_csanv.pkl', 'CSANV')

    # 3. Training SVM (3 Varian)
    train_svm('data/processed/data_training_ori.csv', 'assets/svm/model_svm_ori.pkl', 'ORI')
    train_svm('data/processed/data_training_cstt.csv', 'assets/svm/model_svm_cstt.pkl', 'CSTT')
    train_svm('data/processed/data_training_csanv.csv', 'assets/svm/model_svm_csanv.pkl', 'CSANV')

    # 4. Training Naive Bayes (3 Varian)
    train_nb('data/processed/data_training_ori.csv', 'assets/naive_bayes/model_nb_ori.pkl', 'ORI')
    train_nb('data/processed/data_training_cstt.csv', 'assets/naive_bayes/model_nb_cstt.pkl', 'CSTT')
    train_nb('data/processed/data_training_csanv.csv', 'assets/naive_bayes/model_nb_csanv.pkl', 'CSANV')

    # 5. Evaluasi & Comparison
    run_comparison()

if __name__ == '__main__':
    main()