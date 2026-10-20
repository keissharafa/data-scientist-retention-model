from src.preparation.preprocessing import run_preprocessing
from src.preparation.splitter import run_splitting
from src.preparation.transformation import run_transformation
from src.preparation.selection import run_feature_selection

# 1. Decision Tree
from src.modeling.decision_tree.train import train_model
from src.modeling.decision_tree.evaluate import evaluate_model

# 2. Support Vector Machine (SVM)
from src.modeling.svm.train import train_model_svm
from src.modeling.svm.evaluate import evaluate_model_svm

# 3. Naïve Bayes
from src.modeling.naive_bayes.train import train_naive_bayes
from src.modeling.naive_bayes.evaluate import evaluate_naive_bayes

if __name__ == '__main__':
    print("=" * 75)
    print("RUNNING END-TO-END MACHINE LEARNING PIPELINE".center(75))
    print("=" * 75 + "\n")

    # --------------------------------------------------------------------------
    # FASE 1: PREPARATION & PREPROCESSING DATA
    # --------------------------------------------------------------------------
    run_preprocessing()
    X_train, X_test, y_train, y_test = run_splitting()
    X_train_scaled, X_test_scaled = run_transformation(X_train, X_test)
    run_feature_selection(X_train_scaled, X_test_scaled, y_train, y_test)

    # --------------------------------------------------------------------------
    # FASE 2: MODELING & EVALUASI BENCHMARK
    # --------------------------------------------------------------------------
    # 1. Decision Tree
    print("\n" + "#" * 75)
    print(" MODEL 1: DECISION TREE ".center(75, '#'))
    print("#" * 75 + "\n")
    train_model()
    evaluate_model()

    # 2. Support Vector Machine (SVM)
    print("\n" + "#" * 75)
    print(" MODEL 2: SUPPORT VECTOR MACHINE (SVM) ".center(75, '#'))
    print("#" * 75 + "\n")
    train_model_svm()
    evaluate_model_svm()

    # 3. Naïve Bayes
    print("\n" + "#" * 75)
    print(" MODEL 3: NAÏVE BAYES ".center(75, '#'))
    print("#" * 75 + "\n")
    train_naive_bayes()
    evaluate_naive_bayes()

    print("=" * 75)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!".center(75))
    print("=" * 75)