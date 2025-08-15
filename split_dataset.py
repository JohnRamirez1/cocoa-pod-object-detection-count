import os
import random
import shutil

# Ruta base del dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "DATASET", "cocoa")

# Proporciones
train_ratio = 0.7
val_ratio = 0.2

images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")

# Crear carpetas destino
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(dataset_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split, "labels"), exist_ok=True)

# Lista de imágenes
all_images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))]
random.shuffle(all_images)

# Calcular tamaños
total = len(all_images)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]

def move_files(file_list, split):
    for img_file in file_list:
        name, _ = os.path.splitext(img_file)
        src_img = os.path.join(images_dir, img_file)
        src_label = os.path.join(labels_dir, f"{name}.txt")
        dst_img = os.path.join(dataset_dir, split, "images", img_file)
        dst_label = os.path.join(dataset_dir, split, "labels", f"{name}.txt")

        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

# Mover archivos
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print(f"✅ Train: {len(train_files)}\n✅ Val: {len(val_files)}\n✅ Test: {len(test_files)}")
