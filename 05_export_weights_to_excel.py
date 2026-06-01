import torch
import pandas as pd
import numpy as np

# load model state dict
state_dict = torch.load('toll_road_cnn.pth')

# buat file excel
excel_path = 'model_weights.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for name, param in state_dict.items():
        # ubah tensor ke numpy array
        arr = param.numpy()
        
        # konversi bentuk tensor agar bisa masuk tabel 2D
        if arr.ndim == 1:
            # bias (1D) -> buat jadi tabel 1 kolom
            df = pd.DataFrame(arr, columns=['bias'])
        elif arr.ndim == 2:
            # linear layer weights (2D) -> simpan apa adanya
            df = pd.DataFrame(arr)
        elif arr.ndim == 4:
            # conv layer weights (4D: out_channels, in_channels, height, width)
            # kita reshape jadi 2D: out_channels x (in_channels * height * width)
            sh = arr.shape
            reshaped = arr.reshape(sh[0], -1)
            df = pd.DataFrame(reshaped)
            # beri nama kolom agar jelas
            df.columns = [f'w_flat_{i}' for i in range(reshaped.shape[1])]
        else:
            # tipe dimensi lain
            df = pd.DataFrame(arr.flatten(), columns=['weight'])
            
        # ganti nama sheet agar valid (Excel melarang karakter khusus seperti '.')
        sheet_name = name.replace('.', '_')[:30]
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Selesai! Bobot model berhasil disimpan ke file Excel: '{excel_path}'")
