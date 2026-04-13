import pandas as pd
from Bio import SeqIO
from itertools import product
import random
import os
import numpy as np
from tqdm import tqdm

# --- 1. CONFIGURATION & ALL 64 CODONS ---
K_VALUE = 5
kmers = [''.join(p) for p in product('ATGC', repeat=K_VALUE)]
# 64 Standart Kodon Listesi (AAA, AAC, ... TTT)
standard_codons = [''.join(p) for p in product('ATGC', repeat=3)]

# Kyte-Doolittle & Physics Maps (Sabit tutuldu)
HYDRO_MAP = { 'ATA': 4.5, 'ATC': 4.5, 'ATT': 4.5, 'ATG': 1.9, 'GCA': 1.8, 'GCC': 1.8, 'GAA': -3.5, 'GAG': -3.5, 'GAT': -3.5, 'GAC': -3.5, 'TTC': 2.8, 'TTT': 2.8, 'TTA': 3.8, 'TTG': 3.8, 'CTA': 3.8, 'CTC': 3.8, 'CTG': 3.8, 'CTT': 3.8, 'CCA': -1.6, 'CCC': -1.6, 'CCG': -1.6, 'CCT': -1.6, 'GTA': 4.2, 'GTC': 4.2, 'GGA': -0.4, 'GGC': -0.4, 'GGG': -0.4, 'GGT': -0.4, 'AGC': -0.8, 'AGT': -0.8 }
PHYSICS_MAP = { 'GCT': (89.1, 6.0), 'GCC': (89.1, 6.0), 'GCA': (89.1, 6.0), 'GCG': (89.1, 6.0), 'TGT': (121.2, 5.1), 'TGC': (121.2, 5.1), 'GAT': (133.1, 2.8), 'GAC': (133.1, 2.8), 'GAA': (147.1, 3.2), 'GAG': (147.1, 3.2), 'TTT': (165.2, 5.5), 'TTC': (165.2, 5.5), 'GGT': (75.1, 6.0), 'GGC': (75.1, 6.0), 'GGA': (75.1, 6.0), 'GGG': (75.1, 6.0), 'CAT': (155.2, 7.6), 'CAC': (155.2, 7.6), 'ATT': (131.2, 6.0), 'ATC': (131.2, 6.0), 'ATA': (131.2, 6.0), 'AAA': (146.2, 9.7), 'AAG': (146.2, 9.7), 'ATG': (149.2, 5.7), 'AAT': (132.1, 5.4), 'AAC': (132.1, 5.4), 'CCT': (115.1, 6.3), 'CCC': (115.1, 6.3), 'CAA': (146.1, 5.7), 'CAG': (146.1, 5.7), 'CGT': (174.2, 10.8), 'CGC': (174.2, 10.8), 'TCT': (105.1, 5.7), 'TCC': (105.1, 5.7), 'ACT': (119.1, 5.6), 'ACC': (119.1, 5.6), 'GTT': (117.1, 6.0), 'GTC': (117.1, 6.0), 'TGG': (204.2, 5.9), 'TAT': (181.2, 5.7) }

def extract_universal_features(sequence):
    seq_str = str(sequence).upper()
    seq_len = len(seq_str)
    
    # --- A) K-MER COUNTING (Prefix eklendi) ---
    counts = {f"kmer_{k}": 0 for k in kmers}
    for i in range(seq_len - K_VALUE + 1):
        kmer = seq_str[i:i+K_VALUE]
        key = f"kmer_{kmer}"
        if key in counts: counts[key] += 1
    total_k = sum(counts.values()) or 1
    features = {k: v/total_k for k, v in counts.items()}

    # --- B) CODON FREQUENCY (64 Yeni Özellik) ---
    # Diziyi 3'lü bloklar halinde tarıyoruz
    codon_counts = {f"codon_{c}": 0 for c in standard_codons}
    for i in range(0, seq_len - 2, 3):
        codon = seq_str[i:i+3]
        key = f"codon_{codon}"
        if key in codon_counts: codon_counts[key] += 1
    total_c = sum(codon_counts.values()) or 1
    for k, v in codon_counts.items(): features[k] = v/total_c

    # --- C) BIOMETRICS (CpG, Entropy, Physics) ---
    c, g, cg = seq_str.count('C'), seq_str.count('G'), seq_str.count('CG')
    exp_cg = (c * g) / seq_len if seq_len > 0 else 1
    features['cpg_ratio'] = cg / exp_cg if exp_cg > 0 else 0
    probs = [seq_str.count(b)/seq_len for b in "ATGC"]
    features['entropy'] = -sum(p * np.log2(p) for p in probs if p > 0)

    weights, pIs, hydro_scores = [], [], []
    for i in range(0, seq_len - 2, 3):
        codon = seq_str[i:i+3]
        if codon in PHYSICS_MAP:
            w, p = PHYSICS_MAP[codon]
            weights.append(w); pIs.append(p)
        if codon in HYDRO_MAP:
            hydro_scores.append(HYDRO_MAP[codon])
    
    features['mol_weight_avg'] = sum(weights) / len(weights) if weights else 0
    features['isoelectric_point'] = sum(pIs) / len(pIs) if pIs else 0
    features['hydro_index'] = sum(hydro_scores) / len(hydro_scores) if hydro_scores else 0
    features['gc_ratio'] = (c + g) / seq_len if seq_len > 0 else 0
    features['length'] = seq_len
    
    return features

# --- 2. BIG DATA LOOP ---
data_list = []
classes = ["mammal", "reptile", "bird", "fish"]

print("="*75)
print(f"🚀 Genomic Engine v5.0 - CODON BIAS ENABLED (Big Data Mode)")
print("="*75)

for label in classes:
    file_path = f"{label}_clean.fasta"
    if os.path.exists(file_path):
        records = list(SeqIO.parse(file_path, "fasta"))
        print(f"--- Processing {label.upper()} ({len(records)} records) ---")
        
        for record in tqdm(records):
            if len(record.seq) < 250: continue
            
            # 1. Orijinal Veri
            f = extract_universal_features(record.seq)
            f['target'] = label
            data_list.append(f)
            
            # 2. Augmentation (1 parça yeterli)
            if len(record.seq) > 600:
                start = random.randint(0, len(record.seq) - 400)
                sub_f = extract_universal_features(record.seq[start:start+400])
                sub_f['target'] = label
                data_list.append(sub_f)
    else:
        print(f"❌ Missing: {file_path}")

print("\n📊 Shuffling and Saving Dataset...")
df = pd.DataFrame(data_list).fillna(0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("dna_features.csv", index=False)

print("\n" + "="*75)
print(f"✅ v5.0 COMPLETE! Total Columns: {len(df.columns)}")
print(f"📈 Total Training Samples: {len(df)}")
print("="*75)