import pandas as pd
from Bio import SeqIO
from itertools import product

# 1. Tüm 3-mer kombinasyonlarını oluştur (AAA, AAT... GGG)
kmers = [''.join(p) for p in product('ATGC', repeat=3)]

def kmer_count(sequence, k=3):
    """Dizideki k-mer frekanslarını hesaplar."""
    counts = {kmer: 0 for kmer in kmers}
    for i in range(len(sequence) - k + 1):
        kmer = str(sequence[i:i+k]).upper()
        if kmer in counts:
            counts[kmer] += 1
    
    # Normalizasyon (Uzunluğa bölerek oran bulma)
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}

def gc_content(sequence):
    """GC oranını hesaplar."""
    g = sequence.count("G")
    c = sequence.count("C")
    return (g + c) / len(sequence)

# 2. Verileri işle ve tabloya dönüştür
data_list = []
classes = ["memeli", "surungen", "kus", "balik"]

for label in classes:
    file_path = f"{label}_clean.fasta"
    for record in SeqIO.parse(file_path, "fasta"):
        seq = record.seq
        # k-mer sayımı
        features = kmer_count(seq)
        # Ek özellikler
        features['gc_ratio'] = gc_content(seq)
        features['length'] = len(seq)
        features['target'] = label # Sınıf etiketi
        
        data_list.append(features)

# Pandas DataFrame oluştur ve kaydet
df = pd.DataFrame(data_list)
df.to_csv("dna_features.csv", index=False)
print("Sayısal veri seti 'dna_features.csv' olarak kaydedildi!")
print(f"Toplam Veri Sayısı: {len(df)}")