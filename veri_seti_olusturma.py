import os
import shutil
import random

# Klasörleri oluştur
data_dir = "web_images"
veri_seti_dir = "veri_seti"
train_dir = os.path.join(veri_seti_dir, "train")
val_dir = os.path.join(veri_seti_dir, "val")
test_dir = os.path.join(veri_seti_dir, "test")
normal_dir_train = os.path.join(train_dir, "normal")
normal_dir_val = os.path.join(val_dir, "normal")
normal_dir_test = os.path.join(test_dir, "normal")
cataract_dir_train = os.path.join(train_dir, "cataract")
cataract_dir_val = os.path.join(val_dir, "cataract")
cataract_dir_test = os.path.join(test_dir, "cataract")

# Eğer klasörler yoksa oluştur
for folder in [train_dir, val_dir, test_dir, normal_dir_train, normal_dir_val, normal_dir_test, cataract_dir_train, cataract_dir_val, cataract_dir_test]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Normal ve katarakt gözlere ait tüm dosyaların listesini al
normal_images = os.listdir(os.path.join(data_dir, "normal_images"))
cataract_images = os.listdir(os.path.join(data_dir, "cataract_images"))

# Veri setini karıştır
random.shuffle(normal_images)
random.shuffle(cataract_images)

# Eğitim, doğrulama ve test veri setlerine görselleri kopyala
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Normal gözler için veri setini oluştur
train_split_normal = int(len(normal_images) * train_ratio)
val_split_normal = int(len(normal_images) * (train_ratio + val_ratio))

for img in normal_images[:train_split_normal]:
    shutil.copy(os.path.join(data_dir, "normal_images", img), os.path.join(normal_dir_train, img))

for img in normal_images[train_split_normal:val_split_normal]:
    shutil.copy(os.path.join(data_dir, "normal_images", img), os.path.join(normal_dir_val, img))

for img in normal_images[val_split_normal:]:
    shutil.copy(os.path.join(data_dir, "normal_images", img), os.path.join(normal_dir_test, img))

# Katarakt gözler için veri setini oluştur
train_split_cataract = int(len(cataract_images) * train_ratio)
val_split_cataract = int(len(cataract_images) * (train_ratio + val_ratio))

for img in cataract_images[:train_split_cataract]:
    shutil.copy(os.path.join(data_dir, "cataract_images", img), os.path.join(cataract_dir_train, img))

for img in cataract_images[train_split_cataract:val_split_cataract]:
    shutil.copy(os.path.join(data_dir, "cataract_images", img), os.path.join(cataract_dir_val, img))

for img in cataract_images[val_split_cataract:]:
    shutil.copy(os.path.join(data_dir, "cataract_images", img), os.path.join(cataract_dir_test, img))

print("Veri seti oluşturuldu.")
