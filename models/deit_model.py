import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transformers import DeiTForImageClassification
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve, auc


# Modeli eğitme ve değerlendirme fonksiyonu
def train_and_evaluate_model(model, criterion, optimizer, train_loader, val_loader):
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
        # Scheduler step
        scheduler.step()
        
    return all_train_losses, all_val_losses, all_val_accs, all_val_f1_scores, all_val_recalls, all_val_precisions, all_val_sensitivities, all_val_specificities, all_val_mcc_scores, all_val_aucs, all_val_fpr, all_val_tpr

# Cropped klasörünün tam yolu
cropped_dir = "veri_seti"

kFoldNumber = 5
epochNumber = 30

# Veri yolları
train_data_path = os.path.abspath(os.path.join(cropped_dir, "train"))
val_data_path = os.path.abspath(os.path.join(cropped_dir, "val"))
test_data_path = os.path.abspath(os.path.join(cropped_dir, "test"))

# Veri dönüşümleri
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Veri setini yükleyin
train_dataset = datasets.ImageFolder(root=train_data_path, transform=data_transforms)
val_dataset = datasets.ImageFolder(root=val_data_path, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=test_data_path, transform=data_transforms)


# Optimizasyon fonksiyonu ve kayıp fonksiyonunu tanımlayın
criterion = nn.CrossEntropyLoss()

# K-fold cross-validation için veri setini hazırlama
X = train_dataset.samples
y = [label for _, label in train_dataset.samples]
kfold = KFold(n_splits=kFoldNumber, shuffle=True, random_state=42)


# K-fold cross-validation döngüsü
for fold, (train_index, val_index) in enumerate(kfold.split(X), 1):
    print(f'Fold {fold}/{kFoldNumber}')
    train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_index)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=val_sampler)
     
    # Model yolunu belirleyin
    model_path = "hugging_face/deit-base-distilled-patch16-224"
    # Modeli yerel dizinden yükleyin
    deit_model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224')
    #deit_model = DeiTForImageClassification.from_pretrained(model_path)
    
    num_features = deit_model.classifier.in_features 
    deit_model.classifier = nn.Linear(num_features, 2)
    deit_model.dropout = nn.Dropout(p=0.7)
    optimizer = optim.SGD(deit_model.parameters(), lr=0.00001, momentum=0.99, weight_decay=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #Regulizatör (L2, L1)
    # Optimizasyon (Adam, RMSProp, AdaGrad)
    
    
    # Modeli eğitme ve değerlendirme
    train_losses, val_losses, val_accs, val_f1_scores, val_recalls, val_precisions, val_sensitivities, val_specificities, val_mcc_scores, val_aucs, val_fpr, val_tpr = train_and_evaluate_model(deit_model, criterion, optimizer, train_loader, val_loader)
    
    # Sonuçları bir dosyaya kaydetme
    fold_results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_f1_scores": val_f1_scores,
        "val_recalls": val_recalls,
        "val_precisions": val_precisions,
        "val_sensitivities": val_sensitivities,
        "val_specificities": val_specificities,
        "val_mcc_scores": val_mcc_scores,
        "val_aucs": val_aucs,
        "val_fprs": val_fpr,
        "val_tprs": val_tpr
    }
    
    torch.save(fold_results, f"results/deit/deit{fold}.pth")
    

# Tüm sonuçları birleştirme
all_results = {
    "train_losses": [],
    "val_losses": [],
    "val_accs": [],
    "val_f1_scores": [],
    "val_recalls": [],
    "val_precisions": [],
    "val_sensitivities": [],
    "val_specificities": [],
    "val_mcc_scores": [],
    "val_aucs": [],
    "val_fprs": [],
    "val_tprs": []
}

# Her bir fold sonucunu birleştirme
for fold in range(1, kFoldNumber+1):
    fold_results = torch.load(f"results/deit/deit{fold}.pth")
    for key in all_results:
        all_results[key].extend(fold_results[key])

# Sonuçları bir dosyaya kaydetme
torch.save(all_results, "results/deit/deit_all.pth")