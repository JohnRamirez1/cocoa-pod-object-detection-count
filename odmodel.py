from ultralytics import YOLO
import cv2
import random
import matplotlib.pyplot as plt
import os

# 1️⃣ Cargar modelo preentrenado para transfer learning
model = YOLO("yolov8n.pt")  # Puedes cambiar a yolov8s.pt
# model = YOLO("yolov8s.pt")   # versión nano, más ligera

results_model2 = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="cocoa_detection_v2",
    device=0,
    amp=False,
    patience=15,
    augment=True
)

print("✅ Entrenamiento terminado. Resultados en 'runs/detect/cocoa_detection_v2'")

