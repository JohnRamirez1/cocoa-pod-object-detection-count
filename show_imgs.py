from ultralytics import YOLO
import cv2
import random
import os

# Cargar pesos finales
model = YOLO("runs/detect/cocoa_detection/weights/best.pt")

# Carpeta de validación
val_images_dir = "DATASET/cocoa/val/images"
output_dir = "DATASET/cocoa/val_annotated"
os.makedirs(output_dir, exist_ok=True)

# Seleccionar algunas imágenes de muestra
sample_images = random.sample(os.listdir(val_images_dir), 5)

for img_file in sample_images:
    img_path = os.path.join(val_images_dir, img_file)
    
    # Inferencia
    results = model.predict(img_path)
    
    # Obtener imagen anotada (con bounding boxes)
    annotated_img = results[0].plot()  # devuelve imagen BGR

    # Guardar la imagen anotada
    save_path = os.path.join(output_dir, img_file)
    cv2.imwrite(save_path, annotated_img)

    print(f"Imagen guardada con detecciones: {save_path}")
