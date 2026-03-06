import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib # Modeli kaydetmek için
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Veriyi Yükle
df = pd.read_csv("dna_features.csv")

# 2. Hazırlık
X = df.drop('target', axis=1) # 64 k-mer + gc + length [cite: 15, 16]
y = df['target']             # Memeli, Sürüngen, Kuş, Balık [cite: 4]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Veri Bölme (%80 Eğitim, %20 Test) [cite: 12]
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42)

# 4. Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Modelleri Tanımla [cite: 19]
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

# --- YENİ: Hata Analizi Fonksiyonu ---
def hata_analizi(model, X_test_raw, X_test_scaled, y_test, le, model_name):
    y_pred = model.predict(X_test_scaled)
    hatalar = y_pred != y_test
    
    if np.any(hatalar):
        hata_df = pd.DataFrame({
            'Gercek_Sinif': le.inverse_transform(y_test[hatalar]),
            'Tahmin_Edilen': le.inverse_transform(y_pred[hatalar]),
            'GC_Orani': X_test_raw['gc_ratio'][hatalar],
            'Dizi_Uzunlugu': X_test_raw['length'][hatalar]
        })
        hata_df.to_csv(f"{model_name}_hatalar.csv", index=False)
        print(f"-> {model_name} için {len(hata_df)} hata analiz edildi ve kaydedildi.")
    else:
        print(f"-> {model_name} hatasız tahmin yaptı!")

# --- GÖRSELLEŞTİRME ---
def sonuclari_analiz_et(model, X_test_scaled, y_test, target_names, model_name):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred) [cite: 22]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{model_name} - Karmaşıklık Matrisi')
    plt.savefig(f"{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.show()

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_ [cite: 24]
        indices = np.argsort(importances)[-10:]
        plt.figure(figsize=(10, 5))
        plt.title(f'{model_name} - En Önemli 10 DNA Özelliği')
        plt.barh(range(len(indices)), importances[indices], color='darkgreen')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.tight_layout()
        plt.savefig(f"{model_name.replace(' ', '_')}_feature_importance.png")
        plt.show()

# 7. Döngü: Eğit, Test Et, Analiz Et ve Kaydet
for name, model in models.items():
    print(f"\n=== {name} İşleniyor... ===")
    
    # Cross Validation [cite: 13]
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"CV Ortalama Başarı: %{cv_scores.mean()*100:.2f}")
    
    model.fit(X_train_scaled, y_train)
    
    # Performans Raporu [cite: 22]
    y_pred = model.predict(X_test_scaled)
    print(f"Test Accuracy: %{accuracy_score(y_test, y_pred)*100:.2f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Hata Analizi ve Görselleştirme
    hata_analizi(model, X_test, X_test_scaled, y_test, le, name)
    sonuclari_analiz_et(model, X_test_scaled, y_test, le.classes_, name)

# 8. Modeli ve Araçları Kaydet (predict.py için)
joblib.dump(models["Random Forest"], "dna_model_rf.pkl")
joblib.dump(scaler, "dna_scaler.pkl")
joblib.dump(le, "dna_label_encoder.pkl")

print("\n!!! Modeller ve grafikler kaydedildi. Canlı tahmin için hazırız.")