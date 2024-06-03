import os
import cv2

# Giriş ve çıkış klasörleri
input_folders = ['web_images/normal_images', 'web_images/cataract_images', 'web_images/glokom_images']
output_folders = ['cropped/normal_cropped', 'cropped/cataract_cropped', 'cropped/glokom_cropped']

# Klasörlerin oluşturulması
for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

# Göz tespiti işlevi
def detect_eyes(image):
    # Gri tonlamaya dönüştür
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Göz tespiti yap
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # Gözlerin koordinatlarını döndür
    return eyes

# Kırpma ve kaydetme işlemi
def crop_and_save(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        # Görüntü yolu
        image_path = os.path.join(input_folder, filename)
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is not None:
            # Göz tespiti
            eyes = detect_eyes(image)
            if len(eyes) > 0:
                # En büyük gözü seç
                (x, y, w, h) = max(eyes, key=lambda eye: eye[2] * eye[3])
                # Göz bölgesini kırp
                cropped_image = image[y:y+h, x:x+w]
                # Kırpılmış görüntüyü kaydet
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cropped_image)

# Her bir giriş klasörü için kırpma işlemini yap
for input_folder, output_folder in zip(input_folders, output_folders):
    crop_and_save(input_folder, output_folder)

print("Göz tespiti ve kırpma işlemi tamamlandı.")
