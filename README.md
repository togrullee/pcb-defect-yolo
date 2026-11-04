# PCB Defect Detection with YOLO11

Object detection of six types of printed-circuit-board defects using
[Ultralytics YOLO11](https://docs.ultralytics.com/) (PyTorch), trained on the Kaggle
[PCB Defects](https://www.kaggle.com/datasets/akhatova/pcb-defects) dataset.

| Metric (validation) | Value |
|---|---|
| Precision | 0.956 |
| Recall | 0.969 |
| mAP@50 | **0.979** |
| mAP@50-95 | **0.541** |
| Inference speed (Tesla T4, 640x640 tile) | ~12 ms |

## Dataset

- 693 annotated PCB photographs, ~2200-3000 px per side, Pascal VOC XML annotations.
- 6 classes: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`.
- The Kaggle download also contains `rotation/` and `PCB_USED/` folders; only `images/` and
  `Annotations/` are used.

## What was done

### 1. Environment
YOLO11 (Ultralytics) on PyTorch. A first attempt with TensorFlow/KerasCV was dropped in favour of PyTorch because
Ultralytics YOLO is PyTorch-native.

### 2. Data preparation (`prepare_data.py`)
Defects are small (average ~75 px wide) and the boards are huge. Resizing a whole board to 640 px
would shrink a typical defect to ~15 px, so the images are **tiled** instead:

- Each board is cut into 640x640 tiles at native resolution (stride 480, i.e. 160 px overlap).
- Pascal VOC boxes are converted to YOLO format and clipped to each tile; a box is kept only if at
  least 70% of it lies inside the tile.
- Only tiles containing defects are kept, plus a small number of empty tiles as negatives (15% of
  the positive count per image).
- **Train/val split (80/20, seed 42) is done per source image before tiling**, so overlapping
  tiles of the same board never appear on both sides (no leakage).

Result: 555 train boards -> 4,134 tiles (3,524 defect + 610 empty);
138 val boards -> 1,058 tiles (905 defect + 153 empty).

### 3. Training
- Local CPU benchmarking showed ~0.35 s/tile (a full run would take ~22 h), so training was moved
  to a free Kaggle GPU (Tesla T4) with `pcb_yolo_gpu.ipynb`.
- Model: `yolo11s.pt` (COCO-pretrained, fine-tuned), 640 px, batch 32, up to 60 epochs,
  early stopping with `patience=15`. Best checkpoint chosen by Ultralytics fitness
  (0.1 x mAP50 + 0.9 x mAP50-95) on the validation set.
- `train.py` is the local equivalent and picks settings by hardware (light CPU schedule,
  full GPU schedule).

### 4. Results

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| missing_hole | 0.984 | 1.000 | 0.991 | 0.603 |
| mouse_bite | 0.985 | 0.971 | 0.982 | 0.536 |
| open_circuit | 0.973 | 0.993 | 0.992 | 0.567 |
| short | 0.934 | 0.938 | 0.970 | 0.483 |
| spur | 0.942 | 0.953 | 0.976 | 0.535 |
| spurious_copper | 0.916 | 0.956 | 0.964 | 0.522 |
| **all** | **0.956** | **0.969** | **0.979** | **0.541** |

(1,058 validation tiles, 1,039 defect instances.)

`short` and `spurious_copper` are the weakest classes; `missing_hole` and `open_circuit` the
strongest. The large gap between mAP50 and mAP50-95 means defects are found reliably but boxes
are not always tight to the annotation.

## Limitations (please read)

- **Optimistic estimate.** The dataset consists of only about 10 distinct board designs with
  synthetically added defects, and the split is by image, so validation boards share designs with
  training boards. The 0.979 shows the model finds these defect types on these designs; it does
  **not** show generalisation to unseen board designs, cameras or lighting.
- There is **no separate test set**: the validation set was used for early stopping and checkpoint
  selection as well as for the reported numbers, which biases them slightly upwards.
- Suggested improvements: hold out whole board designs as a test set, add a 3-way split, and
  write a full-board inference script that tiles, predicts and merges boxes (NMS across tiles).

## Usage

```bash
pip install -r requirements.txt

# 1. Download raw data (needs a Kaggle API token)
kaggle datasets download akhatova/pcb-defects -p raw --unzip

# 2. Tile + convert to YOLO format -> dataset/ (+ data.yaml)
python prepare_data.py raw/PCB_DATASET

# 3. Train locally (CPU or CUDA, auto-detected) ...
python train.py
# ... or run pcb_yolo_gpu.ipynb on Kaggle / Colab (GPU) which does all of the above.
```

### Using the trained weights

`weights/pcb_best.pt` (YOLO11s, 19 MB, optimizer state stripped) is included. It was trained on
640x640 tiles, so run it on tiles of that size (a full-board script is future work):

```python
from ultralytics import YOLO
model = YOLO("weights/pcb_best.pt")
results = model.predict("some_640x640_tile.jpg", conf=0.25, imgsz=640)
results[0].show()
```

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | VOC XML -> YOLO labels, tiling, per-image train/val split, `data.yaml` |
| `train.py` | Local training, hardware-dependent configuration |
| `pcb_yolo_gpu.ipynb` | End-to-end Kaggle/Colab GPU notebook (data, tiling, training, evaluation, predictions) |
| `requirements.txt` | Python dependencies |
| `weights/pcb_best.pt` | Trained YOLO11s weights (best epoch by validation fitness) |

The dataset is not redistributed here (its licence is listed as unknown on Kaggle); download it
from Kaggle with the commands above.
