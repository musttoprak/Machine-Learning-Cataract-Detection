import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Veri setini yükle
data = np.load("eye_dataset.npz")
train_data, train_labels = data["train_data"], data["train_labels"]
test_data, test_labels = data["test_data"], data["test_labels"]

# Model oluştur
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 sınıf için çıkış katmanı
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.1)

# Modeli kaydet
model.save("eye_model.h5")

# Eğitilmiş modeli çağır
loaded_model = tf.keras.models.load_model("eye_model.h5")

# Modeli değerlendir
test_loss, test_acc = loaded_model.evaluate(test_data, test_labels)
print("Test doğruluğu:", test_acc)

# Sınıflandırma raporu
predictions = loaded_model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)
print(classification_report(test_labels, predicted_labels))
