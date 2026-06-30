import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
path_gabungan = os.path.join(script_dir, "dataset", "dataset_gabungan.csv")
path_mrl = os.path.join(script_dir, "dataset", "dataset_mrl.csv")

df_gabungan = pd.read_csv(path_gabungan)
df_mrl = pd.read_csv(path_mrl)

print("=== INFO DATASET GABUNGAN (TRAIN) ===")
print(f"Total baris: {len(df_gabungan)}")
print(f"Total kolom: {len(df_gabungan.columns)}")
print("\n=== DISTRIBUSI LABEL GABUNGAN ===")
print(df_gabungan['label'].value_counts())
print("\n=== DISTRIBUSI SOURCE DATASET GABUNGAN ===")
print(df_gabungan['source_dataset'].value_counts())

print("\n" + "="*40)
print("=== INFO DATASET MRL (TEST) ===")
print(f"Total baris: {len(df_mrl)}")
print(f"Total kolom: {len(df_mrl.columns)}")
print("\n=== DISTRIBUSI LABEL MRL ===")
print(df_mrl['label'].value_counts())
print("\n=== DISTRIBUSI SOURCE DATASET MRL ===")
print(df_mrl['source_dataset'].value_counts())

df = df_mrl # fallback for the rest of path checks in the script

print("\n=== CEK PATH GAMBAR ===")
old_base_path = r"E:\sem 4\kecemes\dataset"
new_base_path = os.path.join(script_dir, "dataset")

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

