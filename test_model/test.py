import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Cropped klasörünün tam yolu
cropped_dir = "veri_seti"

kFoldNumber = 3
epochNumber = 50

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

# Google Vit modelini yükleyin
google_vit_model = models.googlenet(pretrained=True)

# Modelin çıktı katmanını değiştirin (Google Vit modelinde 1000 sınıf var, bizim ihtiyacımız ise 2 sınıf)
num_ftrs = google_vit_model.fc.in_features
google_vit_model.fc = nn.Linear(num_ftrs, 2)  # 2 çıkış sınıfı: Normal ve Katarakt

# Optimizasyon fonksiyonu ve kayıp fonksiyonunu tanımlayın
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    google_vit_model.parameters(), 
    lr=0.001,  # Önerilen bir öğrenme oranı kullanabilirsiniz
    momentum=0.9,  
    weight_decay=0.001  
)

# Cross-validation için StratifiedKFold oluşturun
kfold = StratifiedKFold(n_splits=kFoldNumber, shuffle=True, random_state=42)

# Modeli eğitim ve doğrulama fonksiyonları
def train_model(model, criterion, optimizer, train_loader, val_loader):
    train_losses = []
    val_losses = []
    val_accuracies = []
    f1_scores = []
    recalls = []
    precisions = []
    mccs = []
    aucs = []
    fprs, tprs, thresholds = [], [], []
    
    for epoch in range(epochNumber):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(val_epoch_loss)  # val_losses listesine değer ekle
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            f1_scores.append(f1)
            recall = recall_score(all_labels, all_predictions, average='weighted')  # average parametresi ekleniyor
            recalls.append(recall)
            precision = precision_score(all_labels, all_predictions, average='weighted')  # average parametresi ekleniyor
            precisions.append(precision)
            mcc = matthews_corrcoef(all_labels, all_predictions)
            mccs.append(mcc)
            auc = roc_auc_score(all_labels, all_predictions)
            aucs.append(auc)
            
            # ROC eğrisi için verileri topla
            fpr, tpr, threshold = roc_curve(all_labels, all_predictions)
            fprs.append(fpr)
            tprs.append(tpr)
            thresholds.append(threshold)
            
        print(f"Epoch {epoch+1}/{epochNumber}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")

    return train_losses, val_losses, val_accuracies, f1_scores, recalls, precisions, mccs, aucs, fprs, tprs, thresholds



# Modelin eğitimi ve cross-validation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
google_vit_model = google_vit_model.to(device)

# Eğitim ve doğrulama sonuçlarını saklamak için listeler oluştur
all_train_losses, all_val_losses, all_val_accuracies, all_f1_scores, all_recalls, all_precisions, all_mccs, all_aucs, all_fprs, all_tprs, all_thresholds = [], [], [], [], [], [], [], [], [], [], []


for fold, (train_index, val_index) in enumerate(kfold.split(train_dataset.samples, train_dataset.targets)):
    print(f"Fold {fold+1} / {kFoldNumber}")
    train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_index)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=val_sampler)
    
    # Modeli yeniden başlatın
    google_vit_model = models.googlenet(pretrained=True) 
    optimizer = optim.SGD(
        google_vit_model.parameters(), 
        lr=0.001,  
        momentum=0.9,  
        weight_decay=0.001  
    )
    
    # Modeli eğit
    results = train_model(google_vit_model, criterion, optimizer, train_loader, val_loader)
    
    all_train_losses.append(results[0])
    all_val_losses.append(results[1])
    all_val_accuracies.append(results[2])
    all_f1_scores.append(results[3])
    all_recalls.append(results[4])
    all_precisions.append(results[5])
    all_mccs.append(results[6])
    all_aucs.append(results[7])
    all_fprs.append(results[8])
    all_tprs.append(results[9])
    all_thresholds.append(results[10])

# Modeli test et
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")

test_model(google_vit_model, test_loader)


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
plt.show()

# ROC eğrisini çiz
plt.figure(figsize=(8, 8))
for i in range(len(all_fprs)):
    fpr = all_fprs[i][0]  # Her bir kıvrım için fpr değerlerini al
    tpr = all_tprs[i][0]  # Her bir kıvrım için tpr değerlerini al
    fpr_interp = np.linspace(0, 1, 100)  # 100 nokta için eşit aralıklı fpr
    tpr_interp = np.interp(fpr_interp, fpr, tpr)
    plt.plot(fpr_interp, tpr_interp, lw=1, alpha=0.3, label=f'ROC fold {i}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()



#Fold 1 / 3
#Epoch 1/50, Train Loss: 3.6646, Val Loss: 0.5337, Val Acc: 0.7602, F1: 0.7842, Recall: 0.7602, Precision: 0.8432, MCC: 0.5720, AUC: 0.7562
#Epoch 2/50, Train Loss: 0.3479, Val Loss: 0.0908, Val Acc: 0.9024, F1: 0.9024, Recall: 0.9024, Precision: 0.9025, MCC: 0.8050, AUC: 0.9024
#Epoch 3/50, Train Loss: 0.0848, Val Loss: 0.0597, Val Acc: 0.9512, F1: 0.9512, Recall: 0.9512, Precision: 0.9513, MCC: 0.9026, AUC: 0.9512
#Epoch 4/50, Train Loss: 0.0588, Val Loss: 0.0533, Val Acc: 0.9472, F1: 0.9471, Recall: 0.9472, Precision: 0.9479, MCC: 0.8950, AUC: 0.9472
#Epoch 5/50, Train Loss: 0.0397, Val Loss: 0.0481, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 6/50, Train Loss: 0.0306, Val Loss: 0.0487, Val Acc: 0.9553, F1: 0.9553, Recall: 0.9553, Precision: 0.9556, MCC: 0.9108, AUC: 0.9553
#Epoch 7/50, Train Loss: 0.0197, Val Loss: 0.0512, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 8/50, Train Loss: 0.0158, Val Loss: 0.0492, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9675, MCC: 0.9350, AUC: 0.9675
#Epoch 9/50, Train Loss: 0.0171, Val Loss: 0.0515, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9593
#Epoch 10/50, Train Loss: 0.0127, Val Loss: 0.0485, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9675, MCC: 0.9350, AUC: 0.9675
#Epoch 11/50, Train Loss: 0.0090, Val Loss: 0.0481, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9675, MCC: 0.9350, AUC: 0.9675
#Epoch 12/50, Train Loss: 0.0113, Val Loss: 0.0482, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9675, MCC: 0.9350, AUC: 0.9675
#Epoch 13/50, Train Loss: 0.0113, Val Loss: 0.0485, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 14/50, Train Loss: 0.0102, Val Loss: 0.0483, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 15/50, Train Loss: 0.0071, Val Loss: 0.0484, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 16/50, Train Loss: 0.0129, Val Loss: 0.0488, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 17/50, Train Loss: 0.0049, Val Loss: 0.0489, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 18/50, Train Loss: 0.0066, Val Loss: 0.0489, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 19/50, Train Loss: 0.0054, Val Loss: 0.0483, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9593, MCC: 0.9187, AUC: 0.9593
#Epoch 20/50, Train Loss: 0.0069, Val Loss: 0.0482, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 21/50, Train Loss: 0.0050, Val Loss: 0.0491, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 22/50, Train Loss: 0.0041, Val Loss: 0.0487, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 23/50, Train Loss: 0.0050, Val Loss: 0.0493, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 24/50, Train Loss: 0.0053, Val Loss: 0.0512, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 25/50, Train Loss: 0.0036, Val Loss: 0.0508, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 26/50, Train Loss: 0.0041, Val Loss: 0.0489, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 27/50, Train Loss: 0.0030, Val Loss: 0.0496, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9675, MCC: 0.9350, AUC: 0.9675
#Epoch 28/50, Train Loss: 0.0060, Val Loss: 0.0502, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 29/50, Train Loss: 0.0033, Val Loss: 0.0514, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 30/50, Train Loss: 0.0036, Val Loss: 0.0505, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 31/50, Train Loss: 0.0032, Val Loss: 0.0523, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 32/50, Train Loss: 0.0031, Val Loss: 0.0513, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 33/50, Train Loss: 0.0042, Val Loss: 0.0515, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 34/50, Train Loss: 0.0027, Val Loss: 0.0503, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 35/50, Train Loss: 0.0035, Val Loss: 0.0502, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 36/50, Train Loss: 0.0025, Val Loss: 0.0517, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 37/50, Train Loss: 0.0033, Val Loss: 0.0518, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 38/50, Train Loss: 0.0039, Val Loss: 0.0533, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 39/50, Train Loss: 0.0026, Val Loss: 0.0538, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9637, MCC: 0.9271, AUC: 0.9634
#Epoch 40/50, Train Loss: 0.0227, Val Loss: 0.0549, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9593
#Epoch 41/50, Train Loss: 0.0037, Val Loss: 0.0514, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9637, MCC: 0.9271, AUC: 0.9634
#Epoch 42/50, Train Loss: 0.0087, Val Loss: 0.0543, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9593
#Epoch 43/50, Train Loss: 0.0071, Val Loss: 0.0551, Val Acc: 0.9553, F1: 0.9553, Recall: 0.9553, Precision: 0.9560, MCC: 0.9113, AUC: 0.9553
#Epoch 44/50, Train Loss: 0.0073, Val Loss: 0.0498, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 45/50, Train Loss: 0.0044, Val Loss: 0.0515, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 46/50, Train Loss: 0.0015, Val Loss: 0.0506, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 47/50, Train Loss: 0.0033, Val Loss: 0.0516, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 48/50, Train Loss: 0.0028, Val Loss: 0.0528, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9675
#Epoch 49/50, Train Loss: 0.0027, Val Loss: 0.0519, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9675, MCC: 0.9350, AUC: 0.9675
#Epoch 50/50, Train Loss: 0.0028, Val Loss: 0.0524, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Fold 2 / 3
#Epoch 1/50, Train Loss: 3.6602, Val Loss: 0.4627, Val Acc: 0.8537, F1: 0.8644, Recall: 0.8537, Precision: 0.8933, MCC: 0.7283, AUC: 0.8521
#Epoch 2/50, Train Loss: 0.3400, Val Loss: 0.0654, Val Acc: 0.9431, F1: 0.9429, Recall: 0.9431, Precision: 0.9473, MCC: 0.8903, AUC: 0.9427
#Epoch 3/50, Train Loss: 0.1180, Val Loss: 0.0470, Val Acc: 0.9431, F1: 0.9430, Recall: 0.9431, Precision: 0.9449, MCC: 0.8880, AUC: 0.9428
#Epoch 4/50, Train Loss: 0.0839, Val Loss: 0.0482, Val Acc: 0.9512, F1: 0.9512, Recall: 0.9512, Precision: 0.9523, MCC: 0.9035, AUC: 0.9514
#Epoch 5/50, Train Loss: 0.0418, Val Loss: 0.0406, Val Acc: 0.9350, F1: 0.9349, Recall: 0.9350, Precision: 0.9354, MCC: 0.8703, AUC: 0.9348
#Epoch 6/50, Train Loss: 0.0348, Val Loss: 0.0399, Val Acc: 0.9553, F1: 0.9553, Recall: 0.9553, Precision: 0.9560, MCC: 0.9113, AUC: 0.9551
#Epoch 7/50, Train Loss: 0.0213, Val Loss: 0.0397, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9595
#Epoch 8/50, Train Loss: 0.0216, Val Loss: 0.0383, Val Acc: 0.9472, F1: 0.9472, Recall: 0.9472, Precision: 0.9472, MCC: 0.8943, AUC: 0.9471
#Epoch 9/50, Train Loss: 0.0157, Val Loss: 0.0392, Val Acc: 0.9472, F1: 0.9472, Recall: 0.9472, Precision: 0.9472, MCC: 0.8943, AUC: 0.9471
#Epoch 10/50, Train Loss: 0.0161, Val Loss: 0.0392, Val Acc: 0.9472, F1: 0.9471, Recall: 0.9472, Precision: 0.9474, MCC: 0.8945, AUC: 0.9471
#Epoch 11/50, Train Loss: 0.0148, Val Loss: 0.0385, Val Acc: 0.9472, F1: 0.9472, Recall: 0.9472, Precision: 0.9472, MCC: 0.8943, AUC: 0.9471
#Epoch 12/50, Train Loss: 0.0159, Val Loss: 0.0391, Val Acc: 0.9553, F1: 0.9553, Recall: 0.9553, Precision: 0.9555, MCC: 0.9108, AUC: 0.9552
#Epoch 13/50, Train Loss: 0.0091, Val Loss: 0.0376, Val Acc: 0.9472, F1: 0.9472, Recall: 0.9472, Precision: 0.9472, MCC: 0.8943, AUC: 0.9471
#Epoch 14/50, Train Loss: 0.0096, Val Loss: 0.0400, Val Acc: 0.9512, F1: 0.9512, Recall: 0.9512, Precision: 0.9523, MCC: 0.9035, AUC: 0.9510
#Epoch 15/50, Train Loss: 0.0075, Val Loss: 0.0364, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9593, MCC: 0.9187, AUC: 0.9593
#Epoch 16/50, Train Loss: 0.0084, Val Loss: 0.0355, Val Acc: 0.9553, F1: 0.9553, Recall: 0.9553, Precision: 0.9553, MCC: 0.9106, AUC: 0.9553
#Epoch 17/50, Train Loss: 0.0109, Val Loss: 0.0362, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 18/50, Train Loss: 0.0048, Val Loss: 0.0358, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9593, MCC: 0.9187, AUC: 0.9593
#Epoch 19/50, Train Loss: 0.0049, Val Loss: 0.0364, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9595, MCC: 0.9188, AUC: 0.9593
#Epoch 20/50, Train Loss: 0.0062, Val Loss: 0.0351, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 21/50, Train Loss: 0.0054, Val Loss: 0.0348, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9593, MCC: 0.9187, AUC: 0.9593
#Epoch 22/50, Train Loss: 0.0076, Val Loss: 0.0374, Val Acc: 0.9553, F1: 0.9553, Recall: 0.9553, Precision: 0.9555, MCC: 0.9108, AUC: 0.9552
#Epoch 23/50, Train Loss: 0.0088, Val Loss: 0.0358, Val Acc: 0.9553, F1: 0.9553, Recall: 0.9553, Precision: 0.9553, MCC: 0.9106, AUC: 0.9553
#Epoch 24/50, Train Loss: 0.0066, Val Loss: 0.0389, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9604, MCC: 0.9198, AUC: 0.9591
#Epoch 25/50, Train Loss: 0.0095, Val Loss: 0.0401, Val Acc: 0.9512, F1: 0.9512, Recall: 0.9512, Precision: 0.9531, MCC: 0.9043, AUC: 0.9510
#Epoch 26/50, Train Loss: 0.0102, Val Loss: 0.0347, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9674
#Epoch 27/50, Train Loss: 0.0070, Val Loss: 0.0350, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9674
#Epoch 28/50, Train Loss: 0.0056, Val Loss: 0.0391, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9592
#Epoch 29/50, Train Loss: 0.0073, Val Loss: 0.0381, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9592
#Epoch 30/50, Train Loss: 0.0128, Val Loss: 0.0344, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9674
#Epoch 31/50, Train Loss: 0.0058, Val Loss: 0.0370, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9592
#Epoch 32/50, Train Loss: 0.0068, Val Loss: 0.0337, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 33/50, Train Loss: 0.0084, Val Loss: 0.0341, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 34/50, Train Loss: 0.0037, Val Loss: 0.0354, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9592
#Epoch 35/50, Train Loss: 0.0044, Val Loss: 0.0372, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9592
#Epoch 36/50, Train Loss: 0.0058, Val Loss: 0.0339, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9675, MCC: 0.9350, AUC: 0.9675
#Epoch 37/50, Train Loss: 0.0033, Val Loss: 0.0335, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9674
#Epoch 38/50, Train Loss: 0.0048, Val Loss: 0.0347, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9592
#Epoch 39/50, Train Loss: 0.0087, Val Loss: 0.0335, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9675, MCC: 0.9350, AUC: 0.9675
#Epoch 40/50, Train Loss: 0.0032, Val Loss: 0.0340, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9634, MCC: 0.9269, AUC: 0.9634
#Epoch 41/50, Train Loss: 0.0046, Val Loss: 0.0361, Val Acc: 0.9634, F1: 0.9634, Recall: 0.9634, Precision: 0.9637, MCC: 0.9271, AUC: 0.9633
#Epoch 42/50, Train Loss: 0.0032, Val Loss: 0.0366, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9592
#Epoch 43/50, Train Loss: 0.0023, Val Loss: 0.0350, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9674
#Epoch 44/50, Train Loss: 0.0024, Val Loss: 0.0344, Val Acc: 0.9715, F1: 0.9715, Recall: 0.9715, Precision: 0.9716, MCC: 0.9431, AUC: 0.9715
#Epoch 45/50, Train Loss: 0.0051, Val Loss: 0.0347, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9674
#Epoch 46/50, Train Loss: 0.0020, Val Loss: 0.0355, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9592
#Epoch 47/50, Train Loss: 0.0024, Val Loss: 0.0358, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9592
#Epoch 48/50, Train Loss: 0.0040, Val Loss: 0.0366, Val Acc: 0.9593, F1: 0.9593, Recall: 0.9593, Precision: 0.9598, MCC: 0.9192, AUC: 0.9592
#Epoch 49/50, Train Loss: 0.0044, Val Loss: 0.0350, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9674
#Epoch 50/50, Train Loss: 0.0036, Val Loss: 0.0343, Val Acc: 0.9675, F1: 0.9675, Recall: 0.9675, Precision: 0.9676, MCC: 0.9351, AUC: 0.9674
#Fold 3 / 3
#Epoch 1/50, Train Loss: 3.6563, Val Loss: 0.5417, Val Acc: 0.7429, F1: 0.7665, Recall: 0.7429, Precision: 0.8203, MCC: 0.5365, AUC: 0.7406
#Epoch 2/50, Train Loss: 0.3167, Val Loss: 0.1111, Val Acc: 0.8776, F1: 0.8771, Recall: 0.8776, Precision: 0.8825, MCC: 0.7600, AUC: 0.8773
#Epoch 3/50, Train Loss: 0.0802, Val Loss: 0.0836, Val Acc: 0.9184, F1: 0.9184, Recall: 0.9184, Precision: 0.9184, MCC: 0.8367, AUC: 0.9184
#Epoch 4/50, Train Loss: 0.0643, Val Loss: 0.0737, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9225
#Epoch 5/50, Train Loss: 0.0266, Val Loss: 0.0716, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9225
#Epoch 6/50, Train Loss: 0.0308, Val Loss: 0.0699, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9225
#Epoch 7/50, Train Loss: 0.0237, Val Loss: 0.0711, Val Acc: 0.9347, F1: 0.9346, Recall: 0.9347, Precision: 0.9357, MCC: 0.8704, AUC: 0.9346
#Epoch 8/50, Train Loss: 0.0212, Val Loss: 0.0717, Val Acc: 0.9347, F1: 0.9346, Recall: 0.9347, Precision: 0.9365, MCC: 0.8712, AUC: 0.9346
#Epoch 9/50, Train Loss: 0.0194, Val Loss: 0.0721, Val Acc: 0.9388, F1: 0.9387, Recall: 0.9388, Precision: 0.9402, MCC: 0.8790, AUC: 0.9387
#Epoch 10/50, Train Loss: 0.0124, Val Loss: 0.0709, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 11/50, Train Loss: 0.0134, Val Loss: 0.0710, Val Acc: 0.9184, F1: 0.9184, Recall: 0.9184, Precision: 0.9184, MCC: 0.8367, AUC: 0.9184
#Epoch 12/50, Train Loss: 0.0068, Val Loss: 0.0712, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 13/50, Train Loss: 0.0084, Val Loss: 0.0727, Val Acc: 0.9347, F1: 0.9346, Recall: 0.9347, Precision: 0.9365, MCC: 0.8712, AUC: 0.9346
#Epoch 14/50, Train Loss: 0.0146, Val Loss: 0.0700, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9225
#Epoch 15/50, Train Loss: 0.0073, Val Loss: 0.0703, Val Acc: 0.9265, F1: 0.9265, Recall: 0.9265, Precision: 0.9265, MCC: 0.8531, AUC: 0.9265
#Epoch 16/50, Train Loss: 0.0059, Val Loss: 0.0718, Val Acc: 0.9388, F1: 0.9387, Recall: 0.9388, Precision: 0.9402, MCC: 0.8790, AUC: 0.9387
#Epoch 17/50, Train Loss: 0.0060, Val Loss: 0.0733, Val Acc: 0.9388, F1: 0.9387, Recall: 0.9388, Precision: 0.9402, MCC: 0.8790, AUC: 0.9387
#Epoch 18/50, Train Loss: 0.0099, Val Loss: 0.0756, Val Acc: 0.9347, F1: 0.9346, Recall: 0.9347, Precision: 0.9357, MCC: 0.8704, AUC: 0.9346
#Epoch 19/50, Train Loss: 0.0107, Val Loss: 0.0751, Val Acc: 0.9347, F1: 0.9346, Recall: 0.9347, Precision: 0.9357, MCC: 0.8704, AUC: 0.9346
#Epoch 20/50, Train Loss: 0.0080, Val Loss: 0.0729, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9225
#Epoch 21/50, Train Loss: 0.0106, Val Loss: 0.0736, Val Acc: 0.9265, F1: 0.9265, Recall: 0.9265, Precision: 0.9265, MCC: 0.8531, AUC: 0.9265
#Epoch 22/50, Train Loss: 0.0049, Val Loss: 0.0743, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 23/50, Train Loss: 0.0068, Val Loss: 0.0777, Val Acc: 0.9306, F1: 0.9305, Recall: 0.9306, Precision: 0.9329, MCC: 0.8635, AUC: 0.9305
#Epoch 24/50, Train Loss: 0.0070, Val Loss: 0.0738, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 25/50, Train Loss: 0.0044, Val Loss: 0.0740, Val Acc: 0.9306, F1: 0.9306, Recall: 0.9306, Precision: 0.9309, MCC: 0.8615, AUC: 0.9306
#Epoch 26/50, Train Loss: 0.0072, Val Loss: 0.0753, Val Acc: 0.9347, F1: 0.9346, Recall: 0.9347, Precision: 0.9365, MCC: 0.8712, AUC: 0.9346
#Epoch 27/50, Train Loss: 0.0088, Val Loss: 0.0729, Val Acc: 0.9265, F1: 0.9265, Recall: 0.9265, Precision: 0.9265, MCC: 0.8531, AUC: 0.9265
#Epoch 28/50, Train Loss: 0.0027, Val Loss: 0.0730, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 29/50, Train Loss: 0.0065, Val Loss: 0.0740, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 30/50, Train Loss: 0.0044, Val Loss: 0.0730, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 31/50, Train Loss: 0.0089, Val Loss: 0.0795, Val Acc: 0.9306, F1: 0.9305, Recall: 0.9306, Precision: 0.9320, MCC: 0.8626, AUC: 0.9305
#Epoch 32/50, Train Loss: 0.0044, Val Loss: 0.0784, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 33/50, Train Loss: 0.0032, Val Loss: 0.0779, Val Acc: 0.9184, F1: 0.9184, Recall: 0.9184, Precision: 0.9184, MCC: 0.8367, AUC: 0.9184
#Epoch 34/50, Train Loss: 0.0024, Val Loss: 0.0790, Val Acc: 0.9265, F1: 0.9265, Recall: 0.9265, Precision: 0.9266, MCC: 0.8532, AUC: 0.9265
#Epoch 35/50, Train Loss: 0.0030, Val Loss: 0.0799, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 36/50, Train Loss: 0.0022, Val Loss: 0.0799, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 37/50, Train Loss: 0.0040, Val Loss: 0.0792, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 38/50, Train Loss: 0.0036, Val Loss: 0.0782, Val Acc: 0.9265, F1: 0.9265, Recall: 0.9265, Precision: 0.9266, MCC: 0.8532, AUC: 0.9265
#Epoch 39/50, Train Loss: 0.0022, Val Loss: 0.0779, Val Acc: 0.9265, F1: 0.9265, Recall: 0.9265, Precision: 0.9266, MCC: 0.8532, AUC: 0.9265
#Epoch 40/50, Train Loss: 0.0104, Val Loss: 0.0802, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 41/50, Train Loss: 0.0025, Val Loss: 0.0805, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 42/50, Train Loss: 0.0019, Val Loss: 0.0807, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 43/50, Train Loss: 0.0039, Val Loss: 0.0803, Val Acc: 0.9265, F1: 0.9265, Recall: 0.9265, Precision: 0.9266, MCC: 0.8532, AUC: 0.9265
#Epoch 44/50, Train Loss: 0.0032, Val Loss: 0.0769, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 45/50, Train Loss: 0.0020, Val Loss: 0.0785, Val Acc: 0.9306, F1: 0.9306, Recall: 0.9306, Precision: 0.9313, MCC: 0.8619, AUC: 0.9305
#Epoch 46/50, Train Loss: 0.0014, Val Loss: 0.0780, Val Acc: 0.9306, F1: 0.9306, Recall: 0.9306, Precision: 0.9313, MCC: 0.8619, AUC: 0.9305
#Epoch 47/50, Train Loss: 0.0013, Val Loss: 0.0774, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9227, MCC: 0.8451, AUC: 0.9224
#Epoch 48/50, Train Loss: 0.0017, Val Loss: 0.0788, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9227, MCC: 0.8451, AUC: 0.9224
#Epoch 49/50, Train Loss: 0.0013, Val Loss: 0.0797, Val Acc: 0.9224, F1: 0.9224, Recall: 0.9224, Precision: 0.9225, MCC: 0.8449, AUC: 0.9224
#Epoch 50/50, Train Loss: 0.0019, Val Loss: 0.0815, Val Acc: 0.9265, F1: 0.9265, Recall: 0.9265, Precision: 0.9275, MCC: 0.8541, AUC: 0.9264
#Test Accuracy: 0.9570