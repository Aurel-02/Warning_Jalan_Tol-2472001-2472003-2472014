import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from scipy.stats import chi2_contingency
import os

BASE_DIR = r"E:\sem 4\kecemes\dataset"
CSV_PATH = os.path.join(BASE_DIR, "dataset_gabungan.csv")

def main():
    print("Membaca dataset...")
    df = pd.read_csv(CSV_PATH)
    
    print("\n--- 1. DESCRIPTIVE STATISTICS ---")
    total_data = len(df)
    counts = df['label'].value_counts()
    props = df['label'].value_counts(normalize=True) * 100
    
    print(f"Total Sampel: {total_data}")
    print("Frekuensi Label:")
    print(counts)
    print("\nPersentase Label (%):")
    print(props)
    
    # Simpan Plot Descriptive
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='label', palette='Set2')
    plt.title('Distribusi Kelas Drowsiness (Bar Chart)')
    plt.ylabel('Jumlah Gambar')
    
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#fc8d62', '#66c2a5'])
    plt.title('Proporsi Kelas Drowsiness (Pie Chart)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "descriptive_plot.png"))
    print("Plot disimpan sebagai 'descriptive_plot.png'.")
    
    print("\n--- 2. INFERENTIAL STATISTICS ---")
    
    # A. One-Sample Z-Test for Proportion
    # Misalkan H0: Proporsi ngantuk (closed) <= 0.30
    # H1: Proporsi ngantuk > 0.30
    count_closed = counts.get('closed', 0)
    nobs = total_data
    value = 0.30
    
    stat, pval = proportions_ztest(count_closed, nobs, value=value, alternative='larger')
    ci_low, ci_upp = proportion_confint(count_closed, nobs, alpha=0.05, method='normal')
    
    print("\n[A] One-Sample Z-Test untuk Proporsi")
    print("Hipotesis:")
    print("H0: p_closed <= 0.30 (Proporsi mengantuk kurang dari atau sama dengan 30%)")
    print("H1: p_closed > 0.30 (Proporsi mengantuk lebih dari 30%)")
    print(f"Z-Statistic: {stat:.4f}")
    print(f"P-Value: {pval:.4e}")
    print(f"95% Confidence Interval untuk Proporsi Mengantuk: [{ci_low:.4f}, {ci_upp:.4f}]")
    
    # B. Chi-Square Test of Independence
    # H0: Label kantuk (open/closed) independen dari sumber dataset
    # H1: Label kantuk bergantung pada sumber dataset
    print("\n[B] Chi-Square Test of Independence (Label vs Source Dataset)")
    contingency_table = pd.crosstab(df['source_dataset'], df['label'])
    print("\nTabel Kontingensi:")
    print(contingency_table)
    
    chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p_chi2:.4e}")
    print(f"Degrees of Freedom: {dof}")

if __name__ == "__main__":
    main()
