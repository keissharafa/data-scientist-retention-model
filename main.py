from src.preparation.preprocessing import run_preprocessing
from src.preparation.splitter import run_splitting
from src.preparation.transformation import run_transformation
from src.preparation.selection import run_feature_selection
from src.modeling.train import train_model
from src.modeling.evaluate import evaluate_model

if __name__ == '__main__':
    print("=" * 75)
    print("RUNNING END-TO-END MACHINE LEARNING PIPELINE".center(75))
    print("=" * 75 + "\n")

    # 1. Preprocessing Data
    run_preprocessing()

    # 2. Split Data
    X_train, X_test, y_train, y_test = run_splitting()

    # 3. Transformasi Min-Max
    X_train_scaled, X_test_scaled = run_transformation(X_train, X_test)

    # 4. Seleksi Fitur
    run_feature_selection(X_train_scaled, X_test_scaled, y_train, y_test)

    # 5. Pelatihan Model Decision Tree
    train_model()

    # 6. Evaluasi Model
    evaluate_model()

    print("=" * 75)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!".center(75))
    print("=" * 75)