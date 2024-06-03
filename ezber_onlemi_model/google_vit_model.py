import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import googlenet, GoogLeNet_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Cropped klasörünün tam yolu
cropped_dir = "veri_seti"

kFoldNumber = 3
epochNumber = 10

# Veri yolları
train_data_path = os.path.abspath(os.path.join(cropped_dir, "train"))
val_data_path = os.path.abspath(os.path.join(cropped_dir, "val"))
test_data_path = os.path.abspath(os.path.join(cropped_dir, "test"))

# Veri dönüşümleri
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Veri setini yükleyin
train_dataset = datasets.ImageFolder(root=train_data_path, transform=data_transforms)
val_dataset = datasets.ImageFolder(root=val_data_path, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=test_data_path, transform=data_transforms)

# Veri yükleyicileri oluşturun
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Dropout içeren değiştirilmiş GoogleNet modelini oluşturma
class ModifiedGoogLeNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedGoogLeNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])  # Orijinal modeli ayıklayıp sadece özellik çıkarıcı kısmı al
        self.dropout = nn.Dropout(p=0.5)  # Dropout katmanı ekle, p=0.5 dropout oranını belirtir
        self.fc = nn.Linear(original_model.fc.in_features, 2)  # Orijinal modelin fc katmanını değiştir

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)  # Dropout'ı ekle
        x = self.fc(x)
        return x

# K-fold cross-validation için veri setini hazırlama
X = train_dataset.samples
y = [label for _, label in train_dataset.samples]
kfold = KFold(n_splits=kFoldNumber, shuffle=True, random_state=42)

# L1 düzenleme parametresi
l1_lambda = 0.001

# Modeli eğitme ve değerlendirme fonksiyonu
def train_and_evaluate_model(model, criterion, optimizer, train_loader, val_loader, l1_lambda):
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
    all_val_fpr = []
    all_val_tpr = []
    
    for epoch in range(epochNumber):
        model.train()
        running_train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            l1_loss = sum(param.abs().sum() for param in model.parameters()) * l1_lambda
            loss += l1_loss
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_train_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_acc /= 100
        all_train_losses.append(train_loss)
        
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_preds.extend(predicted.tolist())
                val_labels.extend(labels.tolist())

            # Metrics calculation
            val_loss = running_val_loss / len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            f1 = f1_score(val_labels, val_preds)
            recall = recall_score(val_labels, val_preds)
            precision = precision_score(val_labels, val_preds)
            tn, fp, fn, tp = confusion_matrix(val_labels, val_preds).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            mcc = matthews_corrcoef(val_labels, val_preds)
            fpr, tpr, _ = roc_curve(val_labels, val_preds)
            auc_score = auc(fpr, tpr)

             # Append metrics to lists
            all_val_losses.append(val_loss)
            all_val_accs.append(val_acc)
            all_val_f1_scores.append(f1)
            all_val_recalls.append(recall)
            all_val_precisions.append(precision)
            all_val_sensitivities.append(sensitivity)
            all_val_specificities.append(specificity)
            all_val_mcc_scores.append(mcc)
            all_val_aucs.append(auc_score)
            all_val_fpr.append(fpr)
            all_val_tpr.append(tpr)

            print(f'Epoch [{epoch+1}/{epochNumber}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, '
                  f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, '
                  f'MCC: {mcc:.4f}, AUC: {auc_score:.4f}')

    return all_train_losses, all_val_losses, all_val_accs, all_val_f1_scores, all_val_recalls, all_val_precisions, all_val_sensitivities, all_val_specificities, all_val_mcc_scores, all_val_aucs, all_val_fpr, all_val_tpr


# K-fold cross-validation sonuçlarını saklamak için listeler
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
all_val_fpr = []
all_val_tpr = []


# K-fold cross-validation döngüsü
for fold, (train_index, val_index) in enumerate(kfold.split(X), 1):
    print(f'Fold {fold}/{kFoldNumber}')
    train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_index)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=val_sampler)
    
    # Yeni bir model oluşturma
    original_googlenet = googlenet(weights=GoogLeNet_Weights.DEFAULT)
    modified_model = ModifiedGoogLeNet(original_googlenet)
    optimizer = optim.SGD(modified_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Modeli eğitme ve değerlendirme
    train_losses, val_losses, val_accs, val_f1_scores, val_recalls, val_precisions, val_sensitivities, val_specificities, val_mcc_scores, val_aucs, val_fpr, val_tpr = train_and_evaluate_model(modified_model, criterion, optimizer, train_loader, val_loader, l1_lambda)
    
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_val_accs.append(val_accs)
    all_val_f1_scores.append(val_f1_scores)
    all_val_recalls.append(val_recalls)
    all_val_precisions.append(val_precisions)
    all_val_sensitivities.append(val_sensitivities)
    all_val_specificities.append(val_specificities)
    all_val_mcc_scores.append(val_mcc_scores)
    all_val_aucs.append(val_aucs)
    all_val_fpr.append(val_fpr)
    all_val_tpr.append(val_tpr)
    
    fold += 1

# Sonuçları bir dosyaya kaydetme
results = {
    "train_losses": all_train_losses,
    "val_losses": all_val_losses,
    "val_accs": all_val_accs,
    "val_f1_scores": all_val_f1_scores,
    "val_recalls": all_val_recalls,
    "val_precisions": all_val_precisions,
    "val_sensitivities": all_val_sensitivities,
    "val_specificities": all_val_specificities,
    "val_mcc_scores": all_val_mcc_scores,
    "val_aucs": all_val_aucs,
    "val_fprs": all_val_fpr,
    "val_tprs": all_val_tpr
}

torch.save(results, "results/google_vit.pth")

# Loss vs Epoch grafiği
plt.figure(figsize=(10, 5))
for i in range(len(all_train_losses)):
    plt.plot(range(1, len(all_train_losses[i]) + 1), all_train_losses[i], label=f'Train Loss - Fold {i+1}')
    plt.plot(range(1, len(all_val_losses[i]) + 1), all_val_losses[i], label=f'Val Loss - Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.savefig("grafikler/google_vit/loss_vs_epoch.png")
plt.show()

# ROC eğrisini çiz
plt.figure(figsize=(8, 8))
for i in range(len(all_val_fpr)):
    fpr = all_val_fpr[i][0]  # Her bir kıvrım için fpr değerlerini al
    tpr = all_val_tpr[i][0]  # Her bir kıvrım için tpr değerlerini al
    fpr_interp = np.linspace(0, 1, 100)  # 100 nokta için eşit aralıklı fpr
    tpr_interp = np.interp(fpr_interp, fpr, tpr)
    plt.plot(fpr_interp, tpr_interp, lw=1, alpha=0.3, label=f'ROC fold {i}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("grafikler/google_vit/roc_curve.png")
plt.show()




#Fold 1/3
#Epoch [1/10], Train Loss: 164.5202, Train Acc: 0.5743, Val Loss: 0.5734, Val Acc: 0.7154, F1 Score: 0.7651, Recall: 1.0000, Precision: 0.6196, Sensitivity: 1.0000, Specificity: 0.4697, MCC: 0.5395, AUC: 0.7348
#Epoch [2/10], Train Loss: 163.6375, Train Acc: 0.7943, Val Loss: 0.4157, Val Acc: 0.8618, F1 Score: 0.8640, Recall: 0.9474, Precision: 0.7941, Sensitivity: 0.9474, Specificity: 0.7879, MCC: 0.7374, AUC: 0.8676
#Epoch [3/10], Train Loss: 162.6492, Train Acc: 0.8554, Val Loss: 0.3270, Val Acc: 0.8984, F1 Score: 0.8963, Recall: 0.9474, Precision: 0.8504, Sensitivity: 0.9474, Specificity: 0.8561, MCC: 0.8017, AUC: 0.9017
#Epoch [4/10], Train Loss: 161.6655, Train Acc: 0.9002, Val Loss: 0.2742, Val Acc: 0.9106, F1 Score: 0.9083, Recall: 0.9561, Precision: 0.8651, Sensitivity: 0.9561, Specificity: 0.8712, MCC: 0.8254, AUC: 0.9137
#Epoch [5/10], Train Loss: 160.6877, Train Acc: 0.9267, Val Loss: 0.2412, Val Acc: 0.9146, F1 Score: 0.9121, Recall: 0.9561, Precision: 0.8720, Sensitivity: 0.9561, Specificity: 0.8788, MCC: 0.8328, AUC: 0.9175
#Epoch [6/10], Train Loss: 159.7901, Train Acc: 0.9206, Val Loss: 0.2140, Val Acc: 0.9228, F1 Score: 0.9205, Recall: 0.9649, Precision: 0.8800, Sensitivity: 0.9649, Specificity: 0.8864, MCC: 0.8491, AUC: 0.9256
#Epoch [7/10], Train Loss: 158.8458, Train Acc: 0.9470, Val Loss: 0.1942, Val Acc: 0.9268, F1 Score: 0.9237, Recall: 0.9561, Precision: 0.8934, Sensitivity: 0.9561, Specificity: 0.9015, MCC: 0.8554, AUC: 0.9288
#Epoch [8/10], Train Loss: 157.9219, Train Acc: 0.9491, Val Loss: 0.1818, Val Acc: 0.9350, F1 Score: 0.9322, Recall: 0.9649, Precision: 0.9016, Sensitivity: 0.9649, Specificity: 0.9091, MCC: 0.8717, AUC: 0.9370
#Epoch [9/10], Train Loss: 157.0005, Train Acc: 0.9715, Val Loss: 0.1890, Val Acc: 0.9431, F1 Score: 0.9412, Recall: 0.9825, Precision: 0.9032, Sensitivity: 0.9825, Specificity: 0.9091, MCC: 0.8892, AUC: 0.9458
#Epoch [10/10], Train Loss: 156.0923, Train Acc: 0.9776, Val Loss: 0.1543, Val Acc: 0.9431, F1 Score: 0.9397, Recall: 0.9561, Precision: 0.9237, Sensitivity: 0.9561, Specificity: 0.9318, MCC: 0.8863, AUC: 0.9440
#Fold 2/3
#Epoch [1/10], Train Loss: 164.4944, Train Acc: 0.6334, Val Loss: 0.5529, Val Acc: 0.8211, F1 Score: 0.8333, Recall: 0.9167, Precision: 0.7639, Sensitivity: 0.9167, Specificity: 0.7302, MCC: 0.6563, AUC: 0.8234
#Epoch [2/10], Train Loss: 163.6480, Train Acc: 0.8024, Val Loss: 0.4234, Val Acc: 0.8618, F1 Score: 0.8731, Recall: 0.9750, Precision: 0.7905, Sensitivity: 0.9750, Specificity: 0.7540, MCC: 0.7443, AUC: 0.8645
#Epoch [3/10], Train Loss: 162.6289, Train Acc: 0.8778, Val Loss: 0.3393, Val Acc: 0.8984, F1 Score: 0.9012, Recall: 0.9500, Precision: 0.8571, Sensitivity: 0.9500, Specificity: 0.8492, MCC: 0.8016, AUC: 0.8996
#Epoch [4/10], Train Loss: 161.6561, Train Acc: 0.9002, Val Loss: 0.3022, Val Acc: 0.8984, F1 Score: 0.9004, Recall: 0.9417, Precision: 0.8626, Sensitivity: 0.9417, Specificity: 0.8571, MCC: 0.8003, AUC: 0.8994
#Epoch [5/10], Train Loss: 160.7061, Train Acc: 0.9185, Val Loss: 0.2769, Val Acc: 0.8821, F1 Score: 0.8835, Recall: 0.9167, Precision: 0.8527, Sensitivity: 0.9167, Specificity: 0.8492, MCC: 0.7666, AUC: 0.8829
#Epoch [6/10], Train Loss: 159.7622, Train Acc: 0.9287, Val Loss: 0.2622, Val Acc: 0.9024, F1 Score: 0.9040, Recall: 0.9417, Precision: 0.8692, Sensitivity: 0.9417, Specificity: 0.8651, MCC: 0.8078, AUC: 0.9034
#Epoch [7/10], Train Loss: 158.8087, Train Acc: 0.9552, Val Loss: 0.2398, Val Acc: 0.9065, F1 Score: 0.9069, Recall: 0.9333, Precision: 0.8819, Sensitivity: 0.9333, Specificity: 0.8810, MCC: 0.8145, AUC: 0.9071
#Epoch [8/10], Train Loss: 157.9102, Train Acc: 0.9735, Val Loss: 0.2393, Val Acc: 0.9065, F1 Score: 0.9084, Recall: 0.9500, Precision: 0.8702, Sensitivity: 0.9500, Specificity: 0.8651, MCC: 0.8166, AUC: 0.9075
#Epoch [9/10], Train Loss: 156.9755, Train Acc: 0.9837, Val Loss: 0.2277, Val Acc: 0.9065, F1 Score: 0.9076, Recall: 0.9417, Precision: 0.8760, Sensitivity: 0.9417, Specificity: 0.8730, MCC: 0.8154, AUC: 0.9073
#Epoch [10/10], Train Loss: 156.0804, Train Acc: 0.9817, Val Loss: 0.2260, Val Acc: 0.9024, F1 Score: 0.9032, Recall: 0.9333, Precision: 0.8750, Sensitivity: 0.9333, Specificity: 0.8730, MCC: 0.8068, AUC: 0.9032
#Fold 3/3
#Epoch [1/10], Train Loss: 164.5048, Train Acc: 0.6057, Val Loss: 0.5398, Val Acc: 0.8735, F1 Score: 0.8905, Recall: 0.9265, Precision: 0.8571, Sensitivity: 0.9265, Specificity: 0.8073, MCC: 0.7444, AUC: 0.8669
#Epoch [2/10], Train Loss: 163.6517, Train Acc: 0.7886, Val Loss: 0.4013, Val Acc: 0.9102, F1 Score: 0.9197, Recall: 0.9265, Precision: 0.9130, Sensitivity: 0.9265, Specificity: 0.8899, MCC: 0.8180, AUC: 0.9082
#Epoch [3/10], Train Loss: 162.6522, Train Acc: 0.8476, Val Loss: 0.3234, Val Acc: 0.9143, F1 Score: 0.9236, Recall: 0.9338, Precision: 0.9137, Sensitivity: 0.9338, Specificity: 0.8899, MCC: 0.8262, AUC: 0.9119
#Epoch [4/10], Train Loss: 161.6986, Train Acc: 0.8760, Val Loss: 0.2788, Val Acc: 0.9143, F1 Score: 0.9231, Recall: 0.9265, Precision: 0.9197, Sensitivity: 0.9265, Specificity: 0.8991, MCC: 0.8263, AUC: 0.9128
#Epoch [5/10], Train Loss: 160.7054, Train Acc: 0.9106, Val Loss: 0.2356, Val Acc: 0.9306, F1 Score: 0.9386, Recall: 0.9559, Precision: 0.9220, Sensitivity: 0.9559, Specificity: 0.8991, MCC: 0.8596, AUC: 0.9275
#Epoch [6/10], Train Loss: 159.7539, Train Acc: 0.9533, Val Loss: 0.2175, Val Acc: 0.9306, F1 Score: 0.9386, Recall: 0.9559, Precision: 0.9220, Sensitivity: 0.9559, Specificity: 0.8991, MCC: 0.8596, AUC: 0.9275
#Epoch [7/10], Train Loss: 158.8333, Train Acc: 0.9492, Val Loss: 0.2061, Val Acc: 0.9388, F1 Score: 0.9455, Recall: 0.9559, Precision: 0.9353, Sensitivity: 0.9559, Specificity: 0.9174, MCC: 0.8760, AUC: 0.9367
#Epoch [8/10], Train Loss: 157.9043, Train Acc: 0.9654, Val Loss: 0.1948, Val Acc: 0.9347, F1 Score: 0.9424, Recall: 0.9632, Precision: 0.9225, Sensitivity: 0.9632, Specificity: 0.8991, MCC: 0.8681, AUC: 0.9312
#Epoch [9/10], Train Loss: 157.0071, Train Acc: 0.9695, Val Loss: 0.1868, Val Acc: 0.9429, F1 Score: 0.9493, Recall: 0.9632, Precision: 0.9357, Sensitivity: 0.9632, Specificity: 0.9174, MCC: 0.8844, AUC: 0.9403
#Epoch [10/10], Train Loss: 156.0912, Train Acc: 0.9817, Val Loss: 0.1879, Val Acc: 0.9429, F1 Score: 0.9496, Recall: 0.9706, Precision: 0.9296, Sensitivity: 0.9706, Specificity: 0.9083, MCC: 0.8848, AUC: 0.9394