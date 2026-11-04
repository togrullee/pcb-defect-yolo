"""Convert the Kaggle PCB defects dataset (Pascal VOC XML) to tiled YOLO format.

Usage: python prepare_data.py <raw_dataset_dir> [out_dir]

Each large image is cut into TILE x TILE crops at native resolution so tiny
defects keep their size. Only tiles containing defects are kept, plus a few
empty ones as negatives. Train/val split is done per SOURCE image (no leakage).
"""
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
TILE = 640
STRIDE = 480          # 160 px overlap so defects on tile borders appear whole in a neighbour
MIN_VISIBLE = 0.7     # keep a box in a tile only if >=70% of it is inside
NEG_PER_POS = 0.15    # empty tiles kept, as fraction of positive tiles per image
VAL_FRACTION = 0.2
SEED = 42


def read_boxes(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASSES:
            print(f"  skipping unknown class {name!r} in {xml_path.name}")
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((CLASSES.index(name), x1, y1, x2, y2))
    return boxes


def starts(length):
    if length <= TILE:
        return [0]
    s = list(range(0, length - TILE + 1, STRIDE))
    if s[-1] != length - TILE:
        s.append(length - TILE)
    return s


def tile_image(img_path, boxes, out_img, out_lbl, rng):
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    pos, neg = [], []
    for ty in starts(H):
        for tx in starts(W):
            tw, th = min(TILE, W - tx), min(TILE, H - ty)
            lines, touched = [], False
            for c, x1, y1, x2, y2 in boxes:
                ix1, iy1, ix2, iy2 = max(x1, tx), max(y1, ty), min(x2, tx + tw), min(y2, ty + th)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                touched = True
                if (ix2 - ix1) * (iy2 - iy1) / ((x2 - x1) * (y2 - y1)) >= MIN_VISIBLE:
                    lines.append(
                        f"{c} {((ix1 + ix2) / 2 - tx) / tw:.6f} {((iy1 + iy2) / 2 - ty) / th:.6f} "
                        f"{(ix2 - ix1) / tw:.6f} {(iy2 - iy1) / th:.6f}"
                    )
            if lines:
                pos.append((tx, ty, tw, th, lines))
            elif not touched:           # fully empty tile (no partial defects either)
                neg.append((tx, ty, tw, th, []))
    keep = pos + rng.sample(neg, min(len(neg), max(1, round(len(pos) * NEG_PER_POS))))
    for tx, ty, tw, th, lines in keep:
        name = f"{img_path.stem}_{tx}_{ty}"
        cv2.imwrite(str(out_img / f"{name}.jpg"), img[ty:ty + th, tx:tx + tw], [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out_lbl / f"{name}.txt").write_text("\n".join(lines))
    return len(pos), len(keep) - len(pos)


def main():
    raw = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "dataset"
    if out.exists():
        shutil.rmtree(out)

    # Kaggle layout has extra rotation/ and PCB_USED/ folders; use only images/ + Annotations/
    img_root = raw / "images" if (raw / "images").is_dir() else raw
    ann_root = raw / "Annotations" if (raw / "Annotations").is_dir() else raw
    images = {p.stem: p for p in img_root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    pairs = [(x, images[x.stem]) for x in sorted(ann_root.rglob("*.xml")) if x.stem in images]
    print(f"Found {len(pairs)} image/annotation pairs")
    if not pairs:
        sys.exit("No pairs found - check the raw dataset path.")

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n_val = int(len(pairs) * VAL_FRACTION)
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    for split, items in splits.items():
        out_img, out_lbl = out / "images" / split, out / "labels" / split
        out_img.mkdir(parents=True), out_lbl.mkdir(parents=True)
        npos = nneg = 0
        for xml_path, img_path in items:
            p, n = tile_image(img_path, read_boxes(xml_path), out_img, out_lbl, rng)
            npos, nneg = npos + p, nneg + n
        print(f"{split}: {len(items)} source images -> {npos} defect tiles + {nneg} empty tiles")

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
