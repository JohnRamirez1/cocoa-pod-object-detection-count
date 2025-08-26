from ultralytics import YOLO
import cv2
import random
import os

# 3️⃣ Cargar pesos finales
model = YOLO("runs/detect/cocoa_detection_v2/weights/best.pt")

# Exportar a TFLite
model.export(format="tflite")


# 4️⃣ Prueba con imágenes aleatorias de validación
val_images_dir = "DATASET/cocoa/val/images"
sample_images = random.sample(os.listdir(val_images_dir), 10)

for img_file in sample_images:
    img_path = os.path.join(val_images_dir, img_file)
    
    # Inference
    results = model.predict(img_path)
    
    # Obtener imagen con bounding boxes
    annotated_img = results[0].plot()  # imagen anotada

    # Redimensionar la imagen (por ejemplo, 50% del tamaño original)
    scale_percent = 20  # porcentaje del tamaño original
    width = int(annotated_img.shape[1] * scale_percent / 100)
    height = int(annotated_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(annotated_img, dim, interpolation=cv2.INTER_AREA)

    # Mostrar imagen redimensionada
    cv2.imshow(f"Predictions", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()