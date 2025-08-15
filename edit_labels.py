import glob
import os

print(os.getcwd())

base_dir = os.path.dirname(os.path.abspath(__file__))  # Carpeta donde está el script
ruta = os.path.join(base_dir,"DATASET", "cocoa", "labels", "**", "*.txt")
print(ruta)
for file in glob.glob(ruta, recursive=True):
    print(file)
    with open(file, "r") as f:
        lines = f.readlines()
    new_lines = ["0 " + " ".join(line.split()[1:]) + "\n" for line in lines]
    with open(file, "w") as f:
        f.writelines(new_lines)