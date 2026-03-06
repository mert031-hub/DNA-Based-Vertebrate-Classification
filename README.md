# DNA Based Vertebrate Classification 🧬

[cite_start]Bu projenin amacı, omurgalı canlılara ait DNA dizilerini (COI geni) kullanarak örneklerin hangi sınıfa ait olduğunu makine öğrenmesi yöntemiyle tahmin etmektir[cite: 3].

## 📋 Proje Özeti
* [cite_start]**Veri Kaynağı:** NCBI GenBank (COI ve mitokondriyal DNA bölgeleri)[cite: 7, 8].
* [cite_start]**Sınıflar:** Memeliler, Sürüngenler, Kuşlar ve Balıklar[cite: 4].
* [cite_start]**Yöntem:** k-mer frekans analizi ($k=3$) ile 64 sayısal özellik çıkarılmıştır[cite: 15, 16].
* [cite_start]**Algoritmalar:** Random Forest ve Support Vector Machine (SVM)[cite: 19].

## 📊 Performans Sonuçları
Model, 8000 örnekten oluşan genişletilmiş veri seti üzerinde test edilmiştir:
* [cite_start]**Accuracy:** %98.65 [cite: 22]
* [cite_start]**Doğrulama:** 5-Fold Cross Validation uygulanmıştır[cite: 13].

## 🛠️ Kurulum
1. `pip install -r requirements.txt`
2. `python index.py` (Veri toplama)
3. `python train.py` (Model eğitimi ve analiz)
4. `python predict.py` (Canlı tahmin)

## 📈 Çıktılar
[cite_start]Proje sonunda Karmaşıklık Matrisi (Confusion Matrix) ve Özellik Önem Analizi (Feature Importance) grafikleri elde edilmektedir[cite: 24].
