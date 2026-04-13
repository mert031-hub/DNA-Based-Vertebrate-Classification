import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# --- 1. VERİ YÜKLEME ---
print("📂 Veri seti yükleniyor...")
try:
    df = pd.read_csv("dna_features.csv")
    
    # BELLEK OPTİMİZASYONU: float64 -> float32 (RAM kullanımını yarıya indirir)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    print(f"✅ Dataset yüklendi (float32 optimize). Toplam satır: {len(df)}")
except FileNotFoundError:
    print("❌ Hata: 'dna_features.csv' bulunamadı!")
    exit()

X = df.drop('target', axis=1)
y = df['target']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- 2. VERİ BÖLME VE ÖLÇEKLENDİRME ---
print("⚖️ Veri bölünüyor ve ölçeklendiriliyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42)

scaler = StandardScaler()
# Scaler sonrası veriyi float32'ye zorluyoruz
X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_test_scaled = scaler.transform(X_test).astype('float32')

# --- 3. ÜÇLÜ MOTOR (BASE ESTIMATORS) ---
# n_jobs kısıtlaması RAM patlamasını önlemek için kritiktir.
base_models = [
    ('rf', RandomForestClassifier(
        n_estimators=250, # 300'den 250'ye çekerek RAM tasarrufu sağladık
        max_depth=40, 
        min_samples_split=2, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=4, # Tüm çekirdekler yerine 4 çekirdek (RAM koruması)
        verbose=0 
    )),
    ('svm', CalibratedClassifierCV(
        LinearSVC(C=1.0, dual=False, max_iter=2500, random_state=42), 
        cv=3
    )),
    ('xgb', XGBClassifier(
        n_estimators=150, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42, 
        eval_metric='mlogloss',
        n_jobs=4
    ))
]

# --- 4. META-MODEL (THE JUDGE) ---
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3, # 5-Fold yerine 3-Fold (RAM hatasını önlemek için en kritik adım)
    passthrough=False,
    n_jobs=2, # Paralel çalışan model sayısını kısıtladık
    verbose=2 
)

# --- 5. EĞİTİM LOOP ---
start_time = time.time()
print("\n" + "="*50)
print("🔥 Triple-Engine Stacking (v7.2) Eğitimi Başlıyor...")
print("🛡️ RAM Koruma Modu Aktif (float32 + CV=3)")
print("="*50)

try:
    stack_model.fit(X_train_scaled, y_train)
    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    print(f"\n✅ Eğitim başarıyla tamamlandı! Geçen süre: {elapsed:.2f} dakika.")
except MemoryError:
    print("\n❌ KRİTİK: Hala RAM yetmiyor. Lütfen bilgisayardaki diğer programları kapatın.")
    exit()
except Exception as e:
    print(f"\n❌ Hata: {e}")
    exit()

# --- 6. TEST VE RAPORLAMA ---
print("\n📊 Değerlendirme yapılıyor...")
y_pred = stack_model.predict(X_test_scaled)
print(f"\n🏆 Stacking Doğruluğu: %{accuracy_score(y_test, y_pred)*100:.2f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- 7. KAYDETME ---
print("\n💾 Kaydediliyor...")
try:
    joblib.dump(stack_model, "dna_stack_model.pkl")
    joblib.dump(scaler, "dna_scaler.pkl")
    joblib.dump(le, "dna_label_encoder.pkl")
    joblib.dump(X.columns.tolist(), "dna_feature_names.pkl")
    
    bg_indices = np.random.choice(X_train_scaled.shape[0], 100, replace=False)
    joblib.dump(X_train_scaled[bg_indices], "shap_background.pkl")
    
    print("\n" + "="*50)
    print("🏆 SİSTEM v7.2 HAZIR!")
    print("="*50)
except Exception as e:
    print(f"❌ Kayıt hatası: {e}")