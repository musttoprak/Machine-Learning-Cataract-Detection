import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# K-Fold sayısı
kFoldNumber = 5

# Tüm fold'ların sonuçlarını yükleme ve birleştirme
all_train_losses = []
all_val_losses = []
all_val_accs = []
all_val_f1_scores = []
all_val_recalls = []
all_val_precisions = []
all_val_sensitivities = []
all_val_specificities = []
all_val_mcc_scores = []
all_val_aucs = []

for fold in range(1, kFoldNumber + 1):
    fold_results = torch.load(f"results/swin/swin_fold{fold}.pth")
    all_train_losses.append(fold_results["train_losses"])
    all_val_losses.append(fold_results["val_losses"])
    all_val_accs.append(fold_results["val_accs"])
    all_val_f1_scores.append(fold_results["val_f1_scores"])
    all_val_recalls.append(fold_results["val_recalls"])
    all_val_precisions.append(fold_results["val_precisions"])
    all_val_sensitivities.append(fold_results["val_sensitivities"])
    all_val_specificities.append(fold_results["val_specificities"])
    all_val_mcc_scores.append(fold_results["val_mcc_scores"])
    all_val_aucs.append(fold_results["val_aucs"])

# Ortalama değerleri hesapla
avg_val_accs = np.mean(all_val_accs)
avg_val_f1_scores = np.mean(all_val_f1_scores)
avg_val_recalls = np.mean(all_val_recalls)
avg_val_precisions = np.mean(all_val_precisions)
avg_val_sensitivities = np.mean(all_val_sensitivities)
avg_val_specificities = np.mean(all_val_specificities)
avg_val_mcc_scores = np.mean(all_val_mcc_scores)
avg_val_aucs = np.mean(all_val_aucs)

# Performans metriklerini tablo formatında göster
performance_metrics = {
    "Metric": ["Accuracy", "F-measure", "Recall", "Precision", "Sensitivity", "Specificity", "MCC", "AUC"],
    "Value": [avg_val_accs, avg_val_f1_scores, avg_val_recalls, avg_val_precisions, avg_val_sensitivities, avg_val_specificities, avg_val_mcc_scores, avg_val_aucs]
}

performance_df = pd.DataFrame(performance_metrics)

# Matplotlib kullanarak tabloyu çizdir
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=performance_df.values, colLabels=performance_df.columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

plt.title("Model Performance Metrics")
plt.show()
