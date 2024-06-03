import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from transformers import BeitImageProcessor, BeitForImageClassification
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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Google Vit modelini yükleyin
beit_model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224') # 'pretrained=True' parametresi kullanılarak önceden eğitilmiş ağırlıklar çağrılıyor

# Modelin çıktı katmanını değiştirin (Google Vit modelinde 1000 sınıf var, bizim ihtiyacımız ise 2 sınıf)
num_features = beit_model.classifier.in_features
num_classes = 2
beit_model.classifier = nn.Linear(num_features, num_classes)  # Modelin çıktı katmanını değiştirme

# Optimizasyon fonksiyonu ve kayıp fonksiyonunu tanımlayın
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    beit_model.parameters(), 
    lr=0.001,  # Önerilen bir öğrenme oranı kullanabilirsiniz
    momentum=0.9,  
    weight_decay=0.001  
)

# K-fold cross-validation için veri setini hazırlama
X = train_dataset.samples
y = [label for _, label in train_dataset.samples]
kfold = KFold(n_splits=2, shuffle=True, random_state=42)

# Modeli eğitme ve değerlendirme fonksiyonu
def train_and_evaluate_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=5):
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
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            _, predicted = torch.max(F.softmax(outputs.logits, dim=1), 1)
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
                loss = criterion(outputs.logits, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(F.softmax(outputs.logits, dim=1), 1)
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=val_sampler)
    
    # Yeni bir model oluşturma
    beit_model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')
    num_features = beit_model.classifier.in_features
    num_classes = 2
    beit_model.classifier = nn.Linear(num_features, num_classes)  # Modelin çıktı katmanını değiştirme
    optimizer = optim.SGD(beit_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    
    # Modeli eğitme ve değerlendirme
    train_losses, val_losses, val_accs, val_preds, val_labels = train_and_evaluate_model(beit_model, criterion, optimizer, train_loader, val_loader)
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