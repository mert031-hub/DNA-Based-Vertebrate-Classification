import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load v3.1 Dataset
try:
    df = pd.read_csv("dna_features.csv")
    print(f"Dataset loaded. Processing heatmap for {len(df)} samples...")
except FileNotFoundError:
    print("Error: 'dna_features.csv' not found. Run 'analysis.py' first.")
    exit()

# 2. Calculate Class-wise Averages
# Exclude non-k-mer features (gc_ratio and length) for the genomic signature map
kmer_means = df.groupby('target').mean().drop(['gc_ratio', 'length'], axis=1)

# 3. Create Professional Heatmap
plt.figure(figsize=(20, 8))
# YlOrRd (Yellow-Orange-Red) shows density perfectly
sns.heatmap(kmer_means, cmap="YlOrRd", cbar_kws={'label': 'Average k-mer Frequency'})

plt.title("Genomic Signatures: Average k-mer (k=4) Frequencies Across Vertebrate Classes", fontsize=16)
plt.xlabel("k-mer Combinations (256 Features)", fontsize=12)
plt.ylabel("Vertebrate Class", fontsize=12)

# Save the high-resolution visualization for GitHub
plt.tight_layout()
plt.savefig("genomic_heatmap_v3.png", dpi=300)
print("Success! Genomic heatmap saved as 'genomic_heatmap_v3.png'.")
plt.show()