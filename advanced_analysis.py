import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. Load v3.1 Pro Assets
try:
    model = joblib.load("dna_model_svm.pkl") # Updated to SVM Pro
    le = joblib.load("dna_label_encoder.pkl")
    df = pd.read_csv("dna_features.csv")
    X = df.drop('target', axis=1)
    
    scaler = joblib.load("dna_scaler.pkl")
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
except FileNotFoundError:
    print("Error: Required v3.1 .pkl or .csv files not found!")
    exit()

# 2. Batch SHAP Analysis
print("Calculating Global SHAP values (Batch Analysis)...")
explainer = shap.Explainer(model.predict_proba, X_scaled_df)

# Analyze first 50 samples for a global overview
shap_values = explainer(X_scaled_df.iloc[:50])

# 3. Global Visualization
# Select class to analyze (e.g., 0: fish, 1: mammal, etc. based on LabelEncoder)
class_idx = 0 
class_name = le.classes_[class_idx].upper()
print(f"Analyzing global features for class: {class_name}")

plt.figure(figsize=(12, 10))

# Select the SHAP values for the specific class
# Structure: [samples, features, classes]
if len(shap_values.shape) == 3:
    current_values = shap_values[:, :, class_idx]
else:
    current_values = shap_values

shap.plots.bar(current_values, max_display=20, show=False)

plt.title(f"Global Genomic Decision Features: {class_name}")
plt.tight_layout()
plt.savefig(f"global_shap_analysis_{class_name.lower()}.png")
print(f"Success! Chart saved as 'global_shap_analysis_{class_name.lower()}.png'")