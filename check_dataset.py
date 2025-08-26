import os, re, glob, cv2, json
from PIL import Image

ROOT = "DATASET/cocoa"
IMG_DIRS = [os.path.join(ROOT, s, "images") for s in ("train","val","test")]
LBL_DIRS = [os.path.join(ROOT, s, "labels") for s in ("train","val","test")]
NC = 1  # número de clases que declaraste en data.yaml

bad = {"corrupt_images": [], "missing_label": [], "empty_label": [],
       "label_oob": [], "label_bad_vals": [], "unmatched_pairs": []}

def is_image_ok(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Error leyendo {path}")
        return False
    try:
        # Reencode para verificar si es válida
        data = cv2.imdecode(cv2.imencode(".jpg", img)[1], cv2.IMREAD_COLOR)
        return data is not None
    except Exception as e:
        print(f"⚠️ Imagen corrupta {path}: {e}")
        return False

def check_label(path, nc):
    """Valida formato YOLO: class x y w h en [0,1], class en [0, nc-1]"""
    try:
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception:
        return "read_error"

    if len(lines) == 0:
        return "empty_label"

    for li, line in enumerate(lines, 1):
        parts = re.split(r"\s+", line)
        if len(parts) != 5:
            return f"bad_cols@line{li}"
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except Exception:
            return f"parse_error@line{li}"

        if not (0 <= cls < nc):
            return f"class_oob@line{li}"
        for vname, v in zip(("x","y","w","h"), (x,y,w,h)):
            if not (0.0 <= v <= 1.0):
                return f"value_oob_{vname}@line{li}"
        if w <= 0 or h <= 0:
            return f"nonpositive_wh@line{li}"

    return "ok"

def scan_split(img_dir, lbl_dir):
    imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    lbls = sorted([f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")])

    img_set = {os.path.splitext(f)[0] for f in imgs}
    lbl_set = {os.path.splitext(f)[0] for f in lbls}

    # Archivos sin pareja
    for name in sorted(img_set - lbl_set):
        bad["missing_label"].append(os.path.join(img_dir, name))
    for name in sorted(lbl_set - img_set):
        bad["unmatched_pairs"].append(os.path.join(lbl_dir, name + ".txt"))

    # Validar imágenes y labels emparejados
    for name in sorted(img_set & lbl_set):
        ip = os.path.join(img_dir, name + ".jpg")
        if not os.path.exists(ip):
            # intenta otras extensiones
            cand = glob.glob(os.path.join(img_dir, name + ".*"))
            if cand:
                ip = cand[0]
        if not is_image_ok(ip):
            bad["corrupt_images"].append(ip)

        lp = os.path.join(lbl_dir, name + ".txt")
        status = check_label(lp, NC)
        if status == "empty_label":
            bad["empty_label"].append(lp)
        elif status != "ok":
            if "class_oob" in status:
                bad["label_oob"].append(f"{lp}::{status}")
            else:
                bad["label_bad_vals"].append(f"{lp}::{status}")

for img_dir, lbl_dir in zip(IMG_DIRS, LBL_DIRS):
    if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
        scan_split(img_dir, lbl_dir)

out = "dataset_issues.json"
with open(out, "w") as f:
    json.dump(bad, f, indent=2)
print(f"Revisión completa. Resumen:")
for k, v in bad.items():
    print(f"  {k}: {len(v)}")
print(f"Detalle guardado en: {out}")

# Opcional: mover imágenes corruptas a quarantine/
qdir = os.path.join(ROOT, "quarantine")
os.makedirs(qdir, exist_ok=True)
for p in bad["corrupt_images"]:
    try:
        base = os.path.basename(p)
        os.rename(p, os.path.join(qdir, base))
        print("Movida a quarantine:", p)
    except Exception:
        pass
