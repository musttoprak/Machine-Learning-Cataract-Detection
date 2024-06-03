import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Veri setini oluşturmak için gereken değişkenler
data = []
labels = []

# Cropped klasöründeki klasörleri ve etiketlerini al
folders = ['cropped/normal_cropped', 'cropped/cataract_cropped']
#folders = ['cropped/normal_cropped', 'cropped/cataract_cropped']
for label, folder in enumerate(folders):
    for filename in os.listdir(folder):
        # Görüntü yolu
        image_path = os.path.join(folder, filename)
        # Görüntüyü yükle
        image = load_img(image_path, target_size=(224, 224))
        # Görüntüyü diziye dönüştür
        image_array = img_to_array(image)
        # Veri ve etiket listelerine ekle
        data.append(image_array)
        labels.append(label)

# Veriyi numpy dizilerine dönüştür
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Eğitim ve test kümelerine ayır
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Veri setini kaydet
np.savez("eye_dataset.npz", train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)

print("Veri seti oluşturuldu ve 'eye_dataset.npz' olarak kaydedildi.")
