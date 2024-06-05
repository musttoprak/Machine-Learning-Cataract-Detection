import numpy as np
import matplotlib.pyplot as plt

# Sample data (Replace with actual data)
epochs = list(range(1, 31))

# Ortalama loss değerlerini hesapla
avg_train_losses = np.mean(train_loss, axis=0)
avg_val_losses = np.mean(val_loss, axis=0)

# Loss vs Epoch grafiği
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()
