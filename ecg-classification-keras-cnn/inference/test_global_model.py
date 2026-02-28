from tensorflow.keras.models import load_model
import numpy as np

print("Loading global model...")

model = load_model("../model/global_model.h5")
print("✅ Global model loaded")

# Dummy ECG input (same shape)
dummy_ecg = np.random.rand(1, 64, 64, 1)

pred = model.predict(dummy_ecg)

print("Prediction:", pred)
print("Predicted class:", np.argmax(pred))