import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# ==========================================
# 1. Configuration
# ==========================================
MODEL_PATH = 'best_model.keras'  # Change if your file is named differently
IMG_SIZE = (224, 224)
THRESHOLD = 0.5  # Sigmoid threshold

print("Building model architecture...")
# We use weights=None so it doesn't waste time downloading ImageNet base weights, 
# because we are about to overwrite them with your trained weights anyway!
base = MobileNetV2(
    weights=None, 
    include_top=False, 
    input_shape=(*IMG_SIZE, 3)
)

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])

print("Loading trained weights...")
# This specifically targets the weights inside your .keras file, ignoring the buggy structure
model.load_weights(MODEL_PATH)
print("Model loaded successfully!")

# ==========================================
# 2. Initialize Webcam
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to quit.")

# ==========================================
# 3. Real-Time Loop
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # -- Preprocessing --
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, IMG_SIZE)
    normalized_frame = resized_frame / 255.0
    input_tensor = np.expand_dims(normalized_frame, axis=0)

    # -- Prediction --
    prediction = model(input_tensor, training=False)
    score = prediction.numpy()[0][0]

    # -- Interpret Result --
    if score > THRESHOLD:
        label = "Smoking"
        color = (0, 0, 255)  # Red 
        confidence = score * 100
    else:
        label = "Not Smoking"
        color = (0, 255, 0)  # Green 
        confidence = (1 - score) * 100

    # -- Display on Screen --
    text = f"{label}: {confidence:.1f}%"
    cv2.rectangle(frame, (10, 10), (350, 60), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    cv2.imshow('Real-Time Smoking Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# 4. Cleanup
# ==========================================
cap.release()
cv2.destroyAllWindows()