import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import tensorflow as tf

# Load the trained model and labels
model_path = "C:\\Users\\srira\\Downloads\\Models\\keras_model.h5"
labels_path = "C:\\Users\\srira\\Downloads\\Models\\labels.txt"

model = tf.keras.models.load_model(model_path)
with open(labels_path, 'r') as f:
    class_names = f.read().splitlines()

# Video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # White background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop hand with padding
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Preprocess for model
        imgInput = cv2.resize(imgWhite, (224, 224))
        imgInput = np.asarray(imgInput, dtype=np.float32).reshape(1, 224, 224, 3)
        imgInput = (imgInput / 127.5) - 1  # Normalize to [-1, 1]

        # Prediction
        prediction = model.predict(imgInput)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        confidence = prediction[0][class_index]

        # Display
        cv2.putText(img, f'{class_name} ({confidence:.2f})', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
