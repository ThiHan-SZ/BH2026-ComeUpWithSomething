import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────
SRC_IMG = Path(r"TrainingData\src_original\images")
SRC_LBL = Path(r"TrainingData\src_original\labels")

VAL_SPLIT = 0.15   # 15% → ~15 images for val, ~86 for train

TRAIN_IMG = Path(rf"TrainingData\train{VAL_SPLIT * 100:03.0f}\images")
TRAIN_LBL = Path(rf"TrainingData\train{VAL_SPLIT * 100:03.0f}\labels")
VAL_IMG   = Path(rf"TrainingData\valid{VAL_SPLIT * 100:03.0f}\images")
VAL_LBL   = Path(rf"TrainingData\valid{VAL_SPLIT * 100:03.0f}\labels")

# class index → name (from your data.yaml)
CLASS_NAMES = {0: "red_barrel", 1: "yellow_barrel"}

SEED      = 42

# ── Helpers (same as before) ────────────────────────────────────
def read_labels(path):
    if not path.exists():
        return []
    labels = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = float(parts[0])
        coords = list(map(float, parts[1:]))
        if len(coords) < 4 or len(coords) % 2 != 0:
            continue
        xs = coords[0::2]; ys = coords[1::2]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        w  = max(xs) - min(xs)
        h  = max(ys) - min(ys)
        labels.append([cls, cx, cy, w, h])
    return labels

def write_labels(path, labels):
    with open(path, "w") as f:
        for l in labels:
            f.write(" ".join(f"{v:.6f}" for v in l) + "\n")

def flip_h(labels):
    return [[cls, 1.0 - cx, cy, w, h] for cls, cx, cy, w, h in labels]

def save_aug(img, labels, stem, img_dir, lbl_dir):
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
    write_labels(lbl_dir / f"{stem}.txt", labels)

# ── Create dirs ─────────────────────────────────────────────────
for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    d.mkdir(parents=True, exist_ok=True)

# ── Collect originals and assign a stratum to each ─────────────
all_images = sorted(
    p for p in list(SRC_IMG.glob("*.jpg")) + list(SRC_IMG.glob("*.png"))
    if not any(tag in p.stem for tag in ["_flip","_bri","_con","_blur","_noise"])
)

def get_stratum(img_path):
    labels = read_labels(SRC_LBL / (img_path.stem + ".txt"))
    classes = set(int(l[0]) for l in labels)
    if not classes:
        return "empty"
    if classes == {0}:
        return "red_only"
    if classes == {1}:
        return "yellow_only"
    return "mixed"   # contains both

# Group by stratum
strata = defaultdict(list)
for p in all_images:
    strata[get_stratum(p)].append(p)

random.seed(SEED)
val_images, trn_images = [], []

print("Stratum counts:")
for name, group in sorted(strata.items()):
    random.shuffle(group)
    n_val = max(1, round(len(group) * VAL_SPLIT))
    val_images.extend(group[:n_val])
    trn_images.extend(group[n_val:])
    print(f"  {name:12s}: {len(group):3d} total → {n_val} val, {len(group)-n_val} train")

print(f"\nFinal split → train: {len(trn_images)}, val: {len(val_images)}")

# ── Copy val (originals only, no augmentation) ──────────────────
for img_path in val_images:
    shutil.copy(img_path, VAL_IMG / img_path.name)
    labels = read_labels(SRC_LBL / (img_path.stem + ".txt"))
    write_labels(VAL_LBL / (img_path.stem + ".txt"), labels)

# ── Augment train set ───────────────────────────────────────────
generated = 0
for img_path in trn_images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    labels = read_labels(SRC_LBL / (img_path.stem + ".txt"))
    stem = img_path.stem

    save_aug(img, labels, stem, TRAIN_IMG, TRAIN_LBL)

    save_aug(cv2.flip(img, 1), flip_h(labels), f"{stem}_flip", TRAIN_IMG, TRAIN_LBL)
    for beta in [-50, -25, 25, 50]:
        save_aug(cv2.convertScaleAbs(img, alpha=1.0, beta=beta),
                 labels, f"{stem}_bri{beta:+d}", TRAIN_IMG, TRAIN_LBL)
    for alpha in [0.75, 1.35]:
        save_aug(cv2.convertScaleAbs(img, alpha=alpha, beta=0),
                 labels, f"{stem}_con{int(alpha*100)}", TRAIN_IMG, TRAIN_LBL)
    save_aug(cv2.GaussianBlur(img, (5, 5), 0), labels, f"{stem}_blur", TRAIN_IMG, TRAIN_LBL)
    noise = np.random.normal(0, 12, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    save_aug(noisy, labels, f"{stem}_noise", TRAIN_IMG, TRAIN_LBL)
    generated += 9

print(f"\nGenerated {generated} augmented pairs.")
print(f"Train total: {len(list(TRAIN_IMG.glob('*.jpg')))}  |  Val total: {len(list(VAL_IMG.glob('*.jpg')))}")