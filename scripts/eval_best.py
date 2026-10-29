import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import segmentation_models_pytorch as smp

from src.config import FINETUNED, METRICS, NUM_CLASSES, SPLITS
from src.train import GasLeakDataset, evaluate


def main():
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    val_loader = DataLoader(GasLeakDataset(splits["val"]), batch_size=8, shuffle=False)
    test_loader = DataLoader(GasLeakDataset(splits["test"]), batch_size=8, shuffle=False)

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    model.load_state_dict(torch.load(FINETUNED, map_location="cpu", weights_only=True))
    model.eval()

    val_metrics = evaluate(model, val_loader, torch.device("cpu"))
    test_metrics = evaluate(model, test_loader, torch.device("cpu"))

    summary = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    summary["val_best"] = val_metrics
    summary["test"] = test_metrics
    summary["checkpoint"] = FINETUNED.name
    summary["val_images"] = len(splits["val"])
    summary["test_images"] = len(splits["test"])
    METRICS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("VAL", json.dumps(val_metrics))
    print("TEST", json.dumps(test_metrics))


if __name__ == "__main__":
    main()
