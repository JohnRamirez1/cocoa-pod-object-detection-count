import os
import random
import shutil

# 📌 Ruta base del dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir,"archive", "DATASET", "cocoa")
# dataset_dir = r"C:\Users\jjram\Documents\DATASET\cocoa"



# Proporciones de división
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Rutas de imágenes y etiquetas originales
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")

# Crear carpetas destino
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(dataset_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split, "labels"), exist_ok=True)

# Lista de imágenes (solo nombres sin extensión)
all_images = [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")]
random.shuffle(all_images)

# Calcular tamaños
total = len(all_images)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]

def copy_files(file_list, split):
    for img_file in file_list:
        # Nombre sin extensión
        name, _ = os.path.splitext(img_file)

        # Rutas de origen
        src_img = os.path.join(images_dir, img_file)
        src_label = os.path.join(labels_dir, f"{name}.txt")

        # Rutas de destino
        dst_img = os.path.join(dataset_dir, split, "images", img_file)
        dst_label = os.path.join(dataset_dir, split, "labels", f"{name}.txt")

        # Copiar imagen
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)

        # Copiar etiqueta si existe
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

# Copiar a carpetas destino
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print(f"✅ Dataset dividido en:\nTrain: {len(train_files)}\nVal: {len(val_files)}\nTest: {len(test_files)}")
