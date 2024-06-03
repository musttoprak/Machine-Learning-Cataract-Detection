import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import googlenet
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Cropped klasörünün tam yolu
cropped_dir = "veri_seti"

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
google_vit_model = googlenet(pretrained=True)  # 'pretrained=True' parametresi kullanılarak önceden eğitilmiş ağırlıklar çağrılıyor

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

# K-fold cross-validation için veri setini hazırlama
X = train_dataset.samples
y = [label for _, label in train_dataset.samples]
kfold = KFold(n_splits=2, shuffle=True, random_state=42)

# Modeli eğitme ve değerlendirme fonksiyonu
def train_and_evaluate_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    all_train_losses = []
    all_val_losses = []
    all_val_accs = []
    all_val_preds = []
    all_val_labels = []
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
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
        
        val_loss = running_val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        all_val_losses.append(val_loss)
        all_val_accs.append(val_acc)
        all_val_preds.extend(val_preds)
        all_val_labels.extend(val_labels)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
    return all_train_losses, all_val_losses, all_val_accs, all_val_preds, all_val_labels


# Cross-validation sonuçlarını saklamak için listeler
all_train_losses = []
all_val_losses = []
all_val_accs = []
all_val_preds = []
all_val_labels = []

# K-fold cross-validation döngüsü
for fold, (train_index, val_index) in enumerate(kfold.split(X), 1):
    print(f'Fold {fold}')
    train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_index)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=val_sampler)
    
    # Yeni bir model oluşturma
    google_vit_model = googlenet(pretrained=True)
    num_ftrs = google_vit_model.fc.in_features
    google_vit_model.fc = nn.Linear(num_ftrs, 2)
    optimizer = optim.SGD(google_vit_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    
    # Modeli eğitme ve değerlendirme
    train_losses, val_losses, val_accs, val_preds, val_labels = train_and_evaluate_model(google_vit_model, criterion, optimizer, train_loader, val_loader)
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_val_accs.append(val_accs)
    all_val_preds.extend(val_preds)
    all_val_labels.extend(val_labels)
    
    fold += 1
    
# Loss vs epoch grafiği
plt.figure(figsize=(10, 5))
for fold, val_losses in enumerate(all_val_losses):
    plt.plot(range(1, len(val_losses) + 1), val_losses, label=f'Fold {fold+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()

# ROC Curve grafiği
fpr, tpr, _ = roc_curve(all_val_labels, all_val_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()




#Fold 1
#Epoch [1/10], Train Loss: 0.6671, Train Acc: 55.98%, Val Loss: 0.6633, Val Acc: 55.56%
#Epoch [2/10], Train Loss: 0.5346, Train Acc: 82.07%, Val Loss: 0.4764, Val Acc: 82.66%
#Epoch [3/10], Train Loss: 0.4302, Train Acc: 86.68%, Val Loss: 0.4103, Val Acc: 81.03%
#Epoch [4/10], Train Loss: 0.3232, Train Acc: 91.58%, Val Loss: 0.3314, Val Acc: 88.08%
#Epoch [5/10], Train Loss: 0.2636, Train Acc: 91.58%, Val Loss: 0.3149, Val Acc: 87.80%
#Epoch [6/10], Train Loss: 0.2307, Train Acc: 93.48%, Val Loss: 0.2824, Val Acc: 89.43%
#Epoch [7/10], Train Loss: 0.1725, Train Acc: 97.01%, Val Loss: 0.2705, Val Acc: 90.79%
#Epoch [8/10], Train Loss: 0.1489, Train Acc: 97.83%, Val Loss: 0.2469, Val Acc: 91.60%
#Epoch [9/10], Train Loss: 0.1067, Train Acc: 98.64%, Val Loss: 0.2522, Val Acc: 90.79%
#Epoch [10/10], Train Loss: 0.1086, Train Acc: 98.64%, Val Loss: 0.2388, Val Acc: 91.06%
#Fold 2
#Epoch [1/10], Train Loss: 0.6731, Train Acc: 57.72%, Val Loss: 0.6207, Val Acc: 60.33%
#Epoch [2/10], Train Loss: 0.5373, Train Acc: 76.15%, Val Loss: 0.4568, Val Acc: 87.23%
#Epoch [3/10], Train Loss: 0.4188, Train Acc: 84.82%, Val Loss: 0.3812, Val Acc: 88.32%
#Epoch [4/10], Train Loss: 0.3465, Train Acc: 89.43%, Val Loss: 0.3261, Val Acc: 89.67%
#Epoch [5/10], Train Loss: 0.2747, Train Acc: 91.87%, Val Loss: 0.2827, Val Acc: 90.76%
#Epoch [6/10], Train Loss: 0.2369, Train Acc: 92.68%, Val Loss: 0.2529, Val Acc: 91.58%
#Epoch [7/10], Train Loss: 0.1892, Train Acc: 97.56%, Val Loss: 0.2293, Val Acc: 92.66%
#Epoch [8/10], Train Loss: 0.1763, Train Acc: 95.12%, Val Loss: 0.2252, Val Acc: 91.85%
#Epoch [9/10], Train Loss: 0.1404, Train Acc: 95.93%, Val Loss: 0.2184, Val Acc: 93.21%
#Epoch [10/10], Train Loss: 0.1324, Train Acc: 98.10%, Val Loss: 0.1961, Val Acc: 93.48%
