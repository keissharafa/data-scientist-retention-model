import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    print(f"Melatih model {model_name} (Versi ORI - 12 Fitur Utuh)...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return [model_name, acc, prec, rec, f1]

def run_baseline_ori():
    print("=========================================================")
    print("   EVALUASI BASELINE 3 MODEL (VERSI ORI TANPA SELEKSI)   ")
    print("=========================================================")
    
    try:
        df = pd.read_csv('data/interim/cleaned_data.csv')
    except FileNotFoundError:
        print("ERROR: File 'data/interim/cleaned_data.csv' tidak ditemukan.")
        return

    # 2. ENCODING (Meniru splitter.py)
    kolom_kategori = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in kolom_kategori:
        df[col] = le.fit_transform(df[col])
        
    # 3. SPLIT DATA (Meniru splitter.py)
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 4. SCALING MIN-MAX (Meniru transformation.py)
    kolom_numerik = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = MinMaxScaler()
    
    # Copy agar tidak ada warning pandas
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[kolom_numerik] = scaler.fit_transform(X_train[kolom_numerik])
    X_test_scaled[kolom_numerik] = scaler.transform(X_test[kolom_numerik])

    print(f"Dataset ORI siap! Jumlah Fitur: {X_train_scaled.shape[1]}")
    print("-" * 50)
    
    # 5. INISIALISASI & EVALUASI 3 MODEL
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM (Linear)": SVC(kernel='linear', random_state=42),
        "Naive Bayes": GaussianNB()
    }
    
    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X_train_scaled, y_train, X_test_scaled, y_test)
        results.append(res)
        
    print("\n=========================================================")
    print("                 RINGKASAN HASIL VERSI ORI                 ")
    print("=========================================================")
    df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
    
    # Format persentase
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        df_results[col] = (df_results[col] * 100).round(2).astype(str) + "%"
        
    print(df_results.to_string(index=False))
    print("=========================================================\n")

if __name__ == "__main__":
    run_baseline_ori()