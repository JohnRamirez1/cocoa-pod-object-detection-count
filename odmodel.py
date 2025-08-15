from ultralytics import YOLO
import cv2
import random
import os

# 1️⃣ Cargar modelo preentrenado para transfer learning
model = YOLO("yolov8n.pt")  # Puedes cambiar a yolov8s.pt

# 2️⃣ Entrenar con tu dataset
results = model.train(
    data="data.yaml",  # archivo YAML
    epochs=50,
    imgsz=640,
    batch=2,
    name="cocoa_detection",
    device=0,            # usa GPU 0
    amp=False            # deshabilita AMP
)

# 3️⃣ Cargar pesos finales
model = YOLO("runs/detect/cocoa_detection/weights/best.pt")

# 4️⃣ Prueba con imágenes aleatorias de validación
val_images_dir = "DATASET/cocoa/val/images"
sample_images = random.sample(os.listdir(val_images_dir), 3)

for img_file in sample_images:
    img_path = os.path.join(val_images_dir, img_file)
    
    # Inference
    results = model.predict(img_path)
    
    # Obtener imagen con bounding boxes
    annotated_img = results[0].plot()  # devuelve imagen anotada

    # Mostrar imagen con OpenCV
    cv2.imshow(f"Predictions - {img_file}", annotated_img)
    cv2.waitKey(0)  # Espera hasta que presiones una tecla
    cv2.destroyAllWindows()
