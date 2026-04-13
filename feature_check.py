import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier

# --- 1. VERİYİ YÜKLE ---
print("📊 Veri seti yükleniyor...")
df = pd.read_csv("dna_features.csv")
X = df.drop('target', axis=1)
y = df['target']

# --- 2. HIZLI BİR ÖN EĞİTİM ---
print("🚀 Özellik önem dereceleri hesaplanıyor (Bu biraz sürebilir)...")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X, y)

# --- 3. ÖNEMLİ ÖZELLİKLERİ SIRALA ---
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# --- 4. ANALİZ VE GÖRSELLEŞTİRME ---
print("\n🏆 EN GÜÇLÜ 20 ÖZELLİK:")
print(feature_importance_df.head(20))

# En zayıf özellikleri de görelim
print("\n📉 EN ZAYIF 10 ÖZELLİK (Bunlar modelin kafasını karıştırıyor olabilir):")
print(feature_importance_df.tail(10))

# Görselleştirme
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['feature'].head(30), feature_importance_df['importance'].head(30), color='skyblue')
plt.xlabel("Önem Skoru")
plt.title("Modelin Karar Vermesini Sağlayan Top 30 Özellik")
plt.gca().invert_yaxis()
plt.savefig("feature_importance_v5.png")
plt.show()

# --- 5. KRİTİK EŞİK BELİRLEME ---
# Önem değeri ortalamanın altında kalanları temizlemek için bir liste alalım
threshold = feature_importance_df['importance'].mean() # Ortalama önem değeri
keep_features = feature_importance_df[feature_importance_df['importance'] > threshold]['feature'].tolist()

print(f"\n✅ Analiz Tamamlandı!")
print(f"📌 Toplam Özellik: {len(feature_names)}")
print(f"📌 Ortalama Üstü (Altın) Özellik Sayısı: {len(keep_features)}")
print(f"📌 Elenmesi Önerilen (Gürültü) Özellik Sayısı: {len(feature_names) - len(keep_features)}")

# Bu listeyi saklayalım, train.py'da kullanacağız
joblib.dump(keep_features, "selected_features.pkl")