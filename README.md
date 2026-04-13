# DNA-Based Vertebrate Classification 🧬

A machine learning project designed to classify vertebrate species into four main classes (Mammals, Reptiles, Birds, and Fish) using DNA sequence analysis.

## 📌 Project Purpose
[cite_start]The goal of this project is to predict the class of vertebrate animals by processing their DNA sequences through machine learning techniques[cite: 3]. [cite_start]The study focuses on four classes: **Mammals, Reptiles, Birds, and Fish**[cite: 4].

## 🧬 Methodology
- [cite_start]**Data Source:** Primary data is obtained from the **NCBI GenBank** database, focusing on mitochondrial DNA and the **COI gene**[cite: 7, 8].
- [cite_start]**Feature Extraction:** DNA sequences are converted into numerical form using **k-mer frequency analysis** with $k=3$[cite: 15].
- [cite_start]**Feature Set:** The model utilizes $4^3 = 64$ k-mer features, supplemented by **GC content** and **sequence length**[cite: 16].
- [cite_start]**Algorithms:** A comparative analysis is performed using **Random Forest** and **Support Vector Machine (SVM)**[cite: 19].

## 📊 Performance & Results
- **Dataset Size:** Scaled up to ~8000 samples for robust generalization.
- **Accuracy:** Reached a high performance of **98.65%** on the test set.
- [cite_start]**Validation:** 5-Fold Cross-Validation was applied to ensure the model's reliability[cite: 13].
- [cite_start]**Metrics:** Evaluated using Accuracy, Precision, Recall, and F1-Score[cite: 22].



## 🛠️ Project Structure
- `index.py`: Automated DNA sequence retrieval from NCBI.
- `analiz.py`: k-mer feature extraction and dataset creation.
- `train.py`: Model training, hyperparameter tuning, and error analysis.
- `predict.py`: Live prediction script for classifying custom DNA sequences.
