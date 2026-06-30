import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Konstanta
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH_MRL = os.path.join(SCRIPT_DIR, "dataset", "dataset_mrl.csv")
OLD_BASE_PATH = r"E:\sem 4\kecemes\dataset"
NEW_BASE_PATH = os.path.join(SCRIPT_DIR, "dataset")
TRAIN_SAMPLE_SIZE = 10000
TEST_SAMPLE_SIZE = 5000
IMG_SIZE = 64 # Ukuran gambar untuk CNN (64x64)

def fix_path(old_path):
    # Mengganti base path lama dengan base path baru
    # Memastikan case-insensitivity karena Windows
    if old_path.lower().startswith(OLD_BASE_PATH.lower()):
        return NEW_BASE_PATH + old_path[len(OLD_BASE_PATH):]
    return old_path

def process_dataframe(df, sample_size, name):
    # Sampling data dengan proporsi label yang seimbang (stratified)
    print(f"\nMengambil sampel sebanyak {sample_size} data untuk {name}...")
    
    # Stratified sampling
    if len(df) > sample_size:
        df_sample, _ = train_test_split(df, train_size=sample_size, stratify=df['label_num'], random_state=42)
    else:
        df_sample = df
        
    print(f"Distribusi label pada sampel {name}:")
    print(df_sample['label'].value_counts())
    
    # Persiapan wadah data
    X_cnn = []
    X_hog = []
    y = []
    
    print(f"Mulai mengekstrak gambar dan fitur HOG untuk {name}...")
    df_sample = df_sample.reset_index(drop=True)
    
    for i in tqdm(range(len(df_sample))):
        img_path = df_sample.loc[i, 'image_path']
        label = df_sample.loc[i, 'label_num']
        
        # Baca gambar dalam mode grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
            
        # 1. Preprocessing untuk CNN: Resize ke 64x64
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # 2. Ekstraksi fitur untuk HOG
        features_hog = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                           
        X_cnn.append(img_resized)
        X_hog.append(features_hog)
        y.append(label)
        
    # Konversi ke numpy array
    X_cnn = np.array(X_cnn)
    X_cnn = (X_cnn.astype('float32') / 255.0).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    X_hog = np.array(X_hog)
    y = np.array(y)
    
    print(f"Selesai! Berhasil mengekstrak {len(y)} gambar untuk {name}.")
    print(f"Bentuk X_cnn: {X_cnn.shape}")
    print(f"Bentuk X_hog: {X_hog.shape}")
    
    return X_cnn, X_hog, y

def main():
    print("Membaca dataset MRL...")
    df = pd.read_csv(CSV_PATH_MRL)
    
    # Memperbaiki path
    df['image_path'] = df['image_path'].apply(fix_path)
    df['label_num'] = df['label'].map({'closed': 1, 'open': 0})
    
    # Memeriksa keberadaan file (sampling 1000 data awal untuk estimasi)
    print("Memeriksa keberadaan file MRL (sampling 1000 data awal untuk estimasi)...")
    valid_paths = [os.path.exists(p) for p in df['image_path'].head(1000)]
    if not all(valid_paths):
        print("Peringatan: Ada file di dataset MRL yang tidak ditemukan di disk! Pastikan path sudah benar.")
        
    # Memisahkan MRL menjadi set training (70%) dan testing (30%) yang disjoint
    print("\nMembagi MRL dataset menjadi Train (70%) dan Test (30%) secara disjoint...")
    df_train, df_test = train_test_split(df, test_size=0.3, stratify=df['label_num'], random_state=42)
    
    # Proses Train subset
    X_cnn_train, X_hog_train, y_train = process_dataframe(df_train, TRAIN_SAMPLE_SIZE, "Train (MRL)")
    
    # Proses Test subset
    X_cnn_test, X_hog_test, y_test = process_dataframe(df_test, TEST_SAMPLE_SIZE, "Test (MRL)")
    
    # Menyimpan data
    print("\nMenyimpan data ke numpy array (.npy)...")
    np.save('X_cnn_train.npy', X_cnn_train)
    np.save('X_hog_train.npy', X_hog_train)
    np.save('y_train.npy', y_train)
    
    np.save('X_cnn_test.npy', X_cnn_test)
    np.save('X_hog_test.npy', X_hog_test)
    np.save('y_test.npy', y_test)
    print("Data engineering selesai. Semua file npy train dan test dari MRL berhasil dibuat.")

if __name__ == "__main__":
    main()
