import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Veri setini yükle
data = np.load("eye_dataset.npz")
test_data, test_labels = data["test_data"], data["test_labels"]

# Eğitilmiş modeli çağır
loaded_model = tf.keras.models.load_model("eye_model.h5")

# Modeli değerlendir
test_loss, test_acc = loaded_model.evaluate(test_data, test_labels)
print("Test doğruluğu:", test_acc)

# Sınıflandırma raporu
predictions = loaded_model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)
print(classification_report(test_labels, predicted_labels))