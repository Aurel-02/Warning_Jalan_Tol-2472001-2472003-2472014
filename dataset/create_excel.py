import pandas as pd
import os

BASE_DIR = r"E:\sem 4\kecemes\dataset"
CSV_PATH = os.path.join(BASE_DIR, "dataset_gabungan.csv")
OUTPUT_EXCEL = os.path.join(BASE_DIR, "Dataset & Analisis.xlsx")

def main():
    print("Membaca dataset...")
    df = pd.read_csv(CSV_PATH)
    
    # Menyiapkan data analisis untuk Sheet 2
    analysis_data = [
        ["LAPORAN ANALISIS STATISTIKA KANTUK PENGEMUDI", ""],
        ["", ""],
        ["1. STATISTIKA DESKRIPTIF", ""],
        ["Total Sampel", "33069"],
        ["Mengantuk (closed)", "17627 (53.30%)"],
        ["Sadar (open)", "15442 (46.70%)"],
        ["", ""],
        ["2. STATISTIKA INFERENSI", ""],
        ["A. One-Sample Z-Test untuk Proporsi", ""],
        ["Tujuan", "Menguji apakah proporsi pengemudi mengantuk > 30%"],
        ["H0", "p <= 0.30"],
        ["H1", "p > 0.30"],
        ["Z-Statistic", "84.9406"],
        ["P-Value", "0.0000"],
        ["Kesimpulan", "Tolak H0. Proporsi mengantuk secara signifikan > 30%."],
        ["", ""],
        ["B. Chi-Square Test of Independence", ""],
        ["Tujuan", "Menguji independensi keadaan mata dengan sumber dataset"],
        ["H0", "Independen (Tidak ada hubungan)"],
        ["H1", "Dependen (Ada hubungan/bias)"],
        ["Chi-Square Statistic", "1079.88"],
        ["P-Value", "8.39E-234"],
        ["Kesimpulan", "Tolak H0. Terdapat hubungan/bias antar sumber kamera (dataset)."]
    ]
    
    df_analysis = pd.DataFrame(analysis_data, columns=["Komponen", "Nilai / Keterangan"])
    
    # Menulis ke Excel
    print("Menyimpan ke Excel...")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Dataset Mentah', index=False)
        df_analysis.to_excel(writer, sheet_name='Analisis Statistika', index=False)
        
    print(f"Berhasil disimpan di {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()
