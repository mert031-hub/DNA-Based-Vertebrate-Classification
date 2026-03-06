import joblib
import pandas as pd
import numpy as np
from itertools import product

# Araçları yükle
model = joblib.load("dna_model_rf.pkl")
scaler = joblib.load("dna_scaler.pkl")
le = joblib.load("dna_label_encoder.pkl")

def kmer_ozellik_cikar(seq, k=3):
    kmers = [''.join(p) for p in product('ATGC', repeat=3)]
    counts = {k: 0 for k in kmers}
    for i in range(len(seq) - k + 1):
        kmer = str(seq[i:i+k]).upper()
        if kmer in counts:
            counts[kmer] += 1
    total = sum(counts.values())
    features = {k: v/total for k, v in counts.items()}
    features['gc_ratio'] = (seq.upper().count('G') + seq.upper().count('C')) / len(seq)
    features['length'] = len(seq)
    return pd.DataFrame([features])

# Canlı Test
test_dna = input("Tahmin edilecek DNA dizisini yapıştırın: ")
features_df = kmer_ozellik_cikar(test_dna)
scaled_features = scaler.transform(features_df)
tahmin = model.predict(scaled_features)
olasilik = model.predict_proba(scaled_features).max()

print(f"\nTahmin: {le.inverse_transform(tahmin)[0].upper()}")
print(f"Güven Oranı: %{olasilik*100:.2f}")