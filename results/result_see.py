import torch

# Dosyayı yükle
results = torch.load("results/swin.pth")

# Sonuçları görüntüle
print(results)