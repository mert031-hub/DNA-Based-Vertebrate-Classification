import os
from Bio import Entrez, SeqIO
import time

Entrez.email = "dmero904@gmail.com"
classes = {"Memeli": "Mammalia", "Surungen": "Reptilia", "Kus": "Aves", "Balik": "Actinopterygii"}

def buyuk_veri_topla():
    for label, taxon in classes.items():
        print(f"\n>> {label} için 2000 veri hedefleniyor...")
        sorgu = f"({taxon}[Organism]) AND COI[Gene] AND 500:800[SLEN]"
        
        # retmax=2000 yaparak sınırı yükselttik
        with Entrez.esearch(db="nucleotide", term=sorgu, retmax=2000) as handle:
            record = Entrez.read(handle)
            ids = record["IdList"]

        print(f"{len(ids)} adet ID alındı. İndiriliyor...")
        
        # Parçalı indirme (Bağlantı kopmalarını önlemek için 100'erli gruplar)
        clean_records = []
        for i in range(0, len(ids), 100):
            batch_ids = ids[i:i+100]
            with Entrez.efetch(db="nucleotide", id=batch_ids, rettype="fasta", retmode="text") as f:
                batch_data = list(SeqIO.parse(f, "fasta"))
                # Temizlik [cite: 12]
                clean_records.extend([r for r in batch_data if "N" not in r.seq.upper()])
            time.sleep(1) # NCBI sunucularını yormayalım
        
        output_name = f"{label.lower()}_clean.fasta"
        SeqIO.write(clean_records, output_name, "fasta")
        print(f"Tamamlandı: {len(clean_records)} temiz veri -> {output_name}")

if __name__ == "__main__":
    buyuk_veri_topla()