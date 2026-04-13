from Bio import Entrez, SeqIO
import os
import time

# --- CONFIGURATION ---
Entrez.email = "dmero904@example.com" # Buraya kendi mailini yaz (NCBI kuralıdır)
CLASSES = {
    "mammal": "Mammalia[Organism] AND COI[Gene]",
    "bird": "Aves[Organism] AND COI[Gene]",
    "reptile": "Reptilia[Organism] AND COI[Gene]",
    "fish": "Actinopterygii[Organism] AND COI[Gene]"
}
LIMIT = 20000 # Her sınıf için hedef veri sayısı

def fetch_data(label, query):
    file_name = f"{label}_clean.fasta"
    print(f"🚀 {label.upper()} verileri indiriliyor (Hedef: {LIMIT})...")
    
    # 1. Search: ID'leri bul
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=LIMIT, idtype="acc")
    record = Entrez.read(handle)
    handle.close()
    
    id_list = record["IdList"]
    print(f"✅ {len(id_list)} adet ID bulundu. Sekanslar çekiliyor...")

    # 2. Fetch: Sekansları indir ve dosyaya yaz
    with open(file_name, "w") as f:
        # NCBI sunucularını yormamak için 500'erli gruplar halinde çekiyoruz
        batch_size = 500
        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i+batch_size]
            try:
                fetch_handle = Entrez.efetch(db="nucleotide", id=batch_ids, rettype="fasta", retmode="text")
                data = fetch_handle.read()
                f.write(data)
                fetch_handle.close()
                print(f"📥 {i + len(batch_ids)} / {len(id_list)} tamamlandı...")
                time.sleep(1) # Banlanmamak için kısa bir bekleme
            except Exception as e:
                print(f"⚠️ Hata oluştu, atlanıyor: {e}")
                continue

    print(f"{file_name} başarıyla oluşturuldu!\n")

if __name__ == "__main__":
    for label, query in CLASSES.items():
        fetch_data(label, query)