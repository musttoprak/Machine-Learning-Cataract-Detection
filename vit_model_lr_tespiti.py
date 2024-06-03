import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models import googlenet
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from itertools import product




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
])

# Veri setini yükleyin
train_dataset = datasets.ImageFolder(root=train_data_path, transform=data_transforms)
val_dataset = datasets.ImageFolder(root=val_data_path, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=test_data_path, transform=data_transforms)

# Veri yükleyicileri oluşturun
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Google Vit modelini yükleyin
google_vit_model = googlenet(pretrained=True)  # 'pretrained=True' parametresi kullanılarak önceden eğitilmiş ağırlıklar çağrılıyor

# Modelin çıktı katmanını değiştirin (Google Vit modelinde 1000 sınıf var, bizim ihtiyacımız ise 2 sınıf)
num_ftrs = google_vit_model.fc.in_features
google_vit_model.fc = nn.Linear(num_ftrs, 2)  # 2 çıkış sınıfı: Normal ve Katarakt

# Optimizasyon fonksiyonu ve kayıp fonksiyonunu tanımlayın
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(google_vit_model.parameters(), lr=0.001)


# Veri yükleyicilerini oluşturun
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Grid search için parametre kombinasyonlarını belirtin
learning_rates = [0.001, 0.01, 0.1]
betas = [(0.9, 0.999), (0.95, 0.99)]
eps_values = [1e-8, 1e-7, 1e-6]
weight_decays = [0, 1e-4, 1e-3]

best_accuracy = 0.0
best_params = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Grid search ile en iyi parametreleri bulun
for lr, beta, eps, weight_decay in product(learning_rates, betas, eps_values, weight_decays):
    # Modeli yeniden başlatın
    google_vit_model =  googlenet(pretrained=True)
    optimizer = optim.Adam(google_vit_model.parameters(), lr=lr, betas=beta, eps=eps, weight_decay=weight_decay)

    # Modeli eğitin
    for epoch in range(10):
        print(f'{epoch+1}/10')
        # Eğitim aşaması
        google_vit_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = google_vit_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Doğrulama aşaması
        google_vit_model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = google_vit_model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        
        # En iyi doğruluğu kontrol edin ve en iyi parametreleri güncelleyin
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                'lr': lr,
                'betas': beta,
                'eps': eps,
                'weight_decay': weight_decay
            }

print("En iyi parametreler:", best_params)
print("En iyi doğruluk:", best_accuracy)

