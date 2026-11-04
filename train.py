from pathlib import Path

import torch
from ultralytics import YOLO

if __name__ == "__main__":
    here = Path(__file__).parent
    gpu = torch.cuda.is_available()

    if gpu:
        # Full run: all tiles, longer schedule, bigger model.
        cfg = dict(model="yolo11s.pt", epochs=100, batch=32, fraction=1.0, device=0, workers=8)
    else:
        # CPU: ~0.35 s/tile -> 40% of tiles is ~10 min/epoch, 20 epochs is ~4 h.
        cfg = dict(model="yolo11n.pt", epochs=20, batch=16, fraction=0.4, device="cpu", workers=4)

    model = YOLO(cfg.pop("model"))
    model.train(
        data=str(here / "dataset" / "data.yaml"),
        imgsz=640,  # tiles are 640 px, so defects stay at native size
        project=str(here / "runs"),
        name="pcb",
        patience=10,
        **cfg,
    )
    model.val()
