import pandas as pd

# Veriyi yükle
df = pd.read_csv("dna_features.csv")

# 1. Genel Yapı
print("--- Veri Seti Özeti ---")
print(f"Satır Sayısı (Örnekler): {df.shape[0]}")
print(f"Sütun Sayısı (Özellikler): {df.shape[1]}") # 64 k-mer + gc_ratio + length + target = 67

# 2. Sınıf Dağılımı (Dengeli miyiz?)
print("\n--- Sınıf Dağılımı ---")
print(df['target'].value_counts())

# 3. Örnek Özellik Bakışı
print("\n--- İlk 5 Satır (İlk 5 k-mer ve Hedef) ---")
print(df.iloc[:5, [0, 1, 2, 3, 4, -3, -2, -1]]) # İlk 5 k-mer, GC, Uzunluk ve Hedef