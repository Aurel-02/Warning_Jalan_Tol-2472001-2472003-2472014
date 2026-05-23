import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import time

def main():
    print("Memuat data hasil ekstraksi HOG...")
    X_hog = np.load('X_hog.npy')
    y = np.load('y.npy')
    
    print(f"Bentuk data: X = {X_hog.shape}, y = {y.shape}")
    
    print("\nMembagi data menjadi set Pelatihan (80%) dan Pengujian (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Distribusi Kelas pada Data Train:")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))
    
    # Model 1: Random Forest
    print("\n" + "="*40)
    print("Melatih Model 1: Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start_time
    print(f"Selesai dalam {rf_time:.2f} detik.")
    
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Akurasi Random Forest: {rf_acc:.4f}")
    
    # Model 2: Linear SVM
    print("\n" + "="*40)
    print("Melatih Model 2: Linear Support Vector Machine...")
    # Menggunakan LinearSVC karena lebih cepat dan memori-efisien untuk fitur berdimensi tinggi dibanding SVC(kernel='rbf')
    svm_model = LinearSVC(dual=False, C=1.0, random_state=42)
    
    start_time = time.time()
    svm_model.fit(X_train, y_train)
    svm_time = time.time() - start_time
    print(f"Selesai dalam {svm_time:.2f} detik.")
    
    svm_pred = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"Akurasi Linear SVM: {svm_acc:.4f}")
    
    # Pilih model terbaik dan simpan
    print("\n" + "="*40)
    if rf_acc > svm_acc:
        print(">> Random Forest dipilih sebagai model terbaik!")
        best_model = rf_model
        best_pred = rf_pred
        model_name = "random_forest_hog.pkl"
    else:
        print(">> Linear SVM dipilih sebagai model terbaik!")
        best_model = svm_model
        best_pred = svm_pred
        model_name = "linear_svm_hog.pkl"
        
    print("\nLaporan Evaluasi Lengkap Model Terbaik:")
    print(classification_report(y_test, best_pred, target_names=['Open (0)', 'Closed (1)']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    
    print(f"\nMenyimpan model terbaik sebagai '{model_name}'...")
    joblib.dump(best_model, model_name)
    print("Selesai!")

if __name__ == "__main__":
    main()
