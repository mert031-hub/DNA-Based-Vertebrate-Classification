import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Not: Bu kodun çalışması için 'train.py' içindeki model ve test verilerine ihtiyaç var.
# Analiz için Random Forest sonuçlarını kullandığımızı varsayalım:

def plot_results(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # 1. Confusion Matrix Görselleştirme
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Sınıf')
    plt.title('Karmaşıklık Matrisi (Confusion Matrix)')
    plt.show()

    # 2. En Önemli 10 k-mer (Sadece Random Forest için)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:] # En önemli 10 özellik
        
        plt.figure(figsize=(10, 6))
        plt.title('En Önemli 10 DNA Özelliği (k-mer)')
        plt.barh(range(len(indices)), importances[indices], color='g', align='center')
        plt.yticks(range(len(indices)), [X_test.columns[i] for i in indices])
        plt.xlabel('Önem Skoru')
        plt.show()

# Bu fonksiyonu train.py dosyanın sonuna ekleyip çağırabilirsin:
# plot_results(rf_model, X_test_scaled, y_test, le.classes_)