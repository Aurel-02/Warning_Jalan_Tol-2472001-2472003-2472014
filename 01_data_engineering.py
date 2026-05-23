import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Konstanta
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "dataset", "dataset_gabungan.csv")
OLD_BASE_PATH = r"E:\sem 4\kecemes\dataset"
NEW_BASE_PATH = os.path.join(SCRIPT_DIR, "dataset")
SAMPLE_SIZE = 15000
IMG_SIZE = 64 # Ukuran gambar untuk CNN (64x64)

def fix_path(old_path):
    # Mengganti base path lama dengan base path baru
    # Memastikan case-insensitivity karena Windows
    if old_path.lower().startswith(OLD_BASE_PATH.lower()):
        return NEW_BASE_PATH + old_path[len(OLD_BASE_PATH):]
    return old_path

def main():
    print("Membaca dataset...")
    df = pd.read_csv(CSV_PATH)
    
    # Memperbaiki path
    df['image_path'] = df['image_path'].apply(fix_path)
    
    # Filter hanya path yang valid (exists di sistem)
    print("Memeriksa keberadaan file (sampling 1000 data awal untuk estimasi)...")
    valid_paths = [os.path.exists(p) for p in df['image_path'].head(1000)]
    if not all(valid_paths):
        print("Peringatan: Ada file yang tidak ditemukan di disk! Pastikan path sudah benar.")
    
    # Sampling 15.000 data dengan proporsi label yang seimbang (stratified)
    print(f"\nMengambil sampel sebanyak {SAMPLE_SIZE} data...")
    # Label encode: closed -> 1, open -> 0 (Bebas, mari kita pakai closed=1, open=0)
    df['label_num'] = df['label'].map({'closed': 1, 'open': 0})
    
    # Stratified sampling
    # Kita pisahkan train dan sisanya, kita hanya ambil train_size=SAMPLE_SIZE
    if len(df) > SAMPLE_SIZE:
        df_sample, _ = train_test_split(df, train_size=SAMPLE_SIZE, stratify=df['label_num'], random_state=42)
    else:
        df_sample = df
        
    print("Distribusi label pada sampel:")
    print(df_sample['label'].value_counts())
    
    # Persiapan wadah data
    X_cnn = []
    X_hog = []
    y = []
    
    print("\nMulai mengekstrak gambar dan fitur HOG...")
    # Reset index untuk iterasi
    df_sample = df_sample.reset_index(drop=True)
    
    for i in tqdm(range(len(df_sample))):
        img_path = df_sample.loc[i, 'image_path']
        label = df_sample.loc[i, 'label_num']
        
        # Baca gambar dalam mode grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # Jika gagal dibaca, skip
            continue
            
        # 1. Preprocessing untuk CNN: Resize ke 64x64
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # 2. Ekstraksi fitur untuk Traditional ML: HOG
        # orientation=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)
        features_hog = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                           
        X_cnn.append(img_resized)
        X_hog.append(features_hog)
        y.append(label)
        
    # Konversi ke numpy array
    X_cnn = np.array(X_cnn)
    # Normalisasi CNN (0-1) dan tambahkan channel dimension agar sesuai untuk Keras (N, 64, 64, 1)
    X_cnn = (X_cnn.astype('float32') / 255.0).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    X_hog = np.array(X_hog)
    y = np.array(y)
    
    print(f"\nSelesai! Berhasil mengekstrak {len(y)} gambar.")
    print(f"Bentuk X_cnn: {X_cnn.shape}")
    print(f"Bentuk X_hog: {X_hog.shape}")
    
    # Menyimpan data
    print("\nMenyimpan data ke numpy array (.npy)...")
    np.save('X_cnn.npy', X_cnn)
    np.save('X_hog.npy', X_hog)
    np.save('y.npy', y)
    print("Data engineering selesai. File 'X_cnn.npy', 'X_hog.npy', dan 'y.npy' berhasil dibuat.")

if __name__ == "__main__":
    main()
