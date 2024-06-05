import numpy as np
import matplotlib.pyplot as plt

# Sample data (Replace with actual data)
epochs = list(range(1, 31))
train_loss = [0.7409, 0.6225, 0.5197, 0.4327, 0.3820, 0.3444, 0.3105, 0.2893, 0.2623, 0.2491, 0.2444, 0.2441, 0.2365, 0.2484, 0.2445, 0.2333, 0.2465, 0.2326, 0.2361, 0.2256, 0.2246, 0.2307, 0.2362, 0.2331, 0.2326, 0.2306, 0.2205, 0.2292, 0.2288, 0.2250]
val_loss = [0.6924, 0.5462, 0.4064, 0.3618, 0.2942, 0.2567, 0.2265, 0.2143, 0.2020, 0.1802, 0.1857, 0.1821, 0.1819, 0.1914, 0.1764, 0.1840, 0.1848, 0.1817, 0.1940, 0.1779, 0.1795, 0.1676, 0.1762, 0.1845, 0.1647, 0.1884, 0.1714, 0.1819, 0.1859, 0.1827]

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
