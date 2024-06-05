import numpy as np
import torch
import matplotlib.pyplot as plt

# Tüm fold'ların sonuçlarını yükleme ve birleştirme
kFoldNumber = 5 # K-Fold sayısı
all_train_losses = []
all_val_losses = []
all_val_fprs = []
all_val_tprs = []

for fold in range(1, kFoldNumber + 1):
    fold_results = torch.load(f"results/google_vit_4/google_vit_fold{fold}.pth")
    all_train_losses.append(fold_results["train_losses"])
    all_val_losses.append(fold_results["val_losses"])
    all_val_fprs.append(fold_results["val_fprs"])
    all_val_tprs.append(fold_results["val_tprs"])

# Ortalama loss değerlerini hesapla
avg_train_losses = np.mean(all_train_losses, axis=0)
avg_val_losses = np.mean(all_val_losses, axis=0)

# Loss vs Epoch grafiği
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Average Train Loss')
plt.plot(range(1, len(avg_val_losses) + 1), avg_val_losses, label='Average Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()

# ROC eğrisini oluşturma
mean_fpr = np.linspace(0, 1, 100)
all_tprs = []

# Her bir fold için fpr ve tpr değerlerini topla
for i in range(len(all_val_fprs)):
    for j in range(len(all_val_fprs[i])):
        fpr = all_val_fprs[i][j]
        tpr = all_val_tprs[i][j]
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        all_tprs.append(tpr_interp)

# Tüm tpr değerlerinin ortalamasını al
mean_tpr = np.mean(all_tprs, axis=0)

# ROC eğrisini çiz
plt.figure(figsize=(8, 8))
plt.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
