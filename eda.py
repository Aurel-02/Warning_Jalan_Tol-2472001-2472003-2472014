import pandas as pd
import os

path = r"E:\SEM 4\KECEMES\Warning_Jalan_Tol-2472001-2472003-2472014\dataset\dataset_gabungan.csv"
df = pd.read_csv(path)

print("=== INFO DATASET ===")
print(f"Total baris: {len(df)}")
print(f"Total kolom: {len(df.columns)}")
print("\n=== DISTRIBUSI LABEL ===")
print(df['label'].value_counts())
print("\n=== DISTRIBUSI SOURCE DATASET ===")
print(df['source_dataset'].value_counts())

print("\n=== CEK PATH GAMBAR ===")
old_base_path = r"E:\sem 4\kecemes\dataset"
new_base_path = r"E:\SEM 4\KECEMES\Warning_Jalan_Tol-2472001-2472003-2472014\dataset"

print("Contoh path asli di csv:")
print(df['image_path'].iloc[0])

# Cek beberapa file dengan path baru
exists_count = 0
for i in range(5):
    old_path = df['image_path'].iloc[i]
    new_path = old_path.replace(old_base_path, new_base_path, 1) # replace ignoring case might be tricky
    new_path = new_path.replace(r"E:\sem 4\kecemes\dataset", new_base_path)
    
    # Just to be safe with case insensitivity
    if old_path.lower().startswith(old_base_path.lower()):
        new_path = new_base_path + old_path[len(old_base_path):]
    else:
        new_path = old_path
        
    print(f"\nCek file ke-{i+1}:")
    print(f"Path Baru: {new_path}")
    print(f"Ada di disk?: {os.path.exists(new_path)}")

