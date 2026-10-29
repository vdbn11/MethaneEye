import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import segmentation_models_pytorch as smp

from src.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_H,
    IMG_W,
    IMAGES_DIR,
    MASKS_DIR,
    METRICS,
    NUM_CLASSES,
    PRETRAINED_V1,
    SPLITS,
    FINETUNED,
)


class GasLeakDataset(Dataset):
    def __init__(self, names, train=False):
        self.names = names
        self.train = train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        stem = Path(name).stem
        image = Image.open(IMAGES_DIR / name).convert("RGB")
        mask = Image.open(MASKS_DIR / (stem + ".png"))

        image = TF.resize(image, [IMG_H, IMG_W], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [IMG_H, IMG_W], interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.25:
                image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
                image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))

        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = torch.from_numpy(np.array(mask, dtype=np.int64))
        return tensor, target


def dice_score(probs, target, mask_classes):
    scores = []
    for cls in mask_classes:
        pred = probs == cls
        truth = target == cls
        intersection = float((pred & truth).sum())
        denominator = float(pred.sum() + truth.sum())
        scores.append((2.0 * intersection + 1e-6) / (denominator + 1e-6))
    return float(np.mean(scores))


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total = 0
    pixel_correct = 0
    pixel_count = 0
    dice_values = []
    iou_values = []
    weight = torch.tensor([0.15, 1.0, 2.5], device=device)

    with torch.no_grad():
        for step, (images, targets) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            total_loss += float(F.cross_entropy(logits, targets, weight=weight)) * images.size(0)
            total += images.size(0)

            predictions = torch.argmax(logits, dim=1)
            pixel_correct += int((predictions == targets).sum())
            pixel_count += targets.numel()

            pred_gas = (predictions > 0).cpu().numpy()
            true_gas = (targets > 0).cpu().numpy()
            intersection = np.logical_and(pred_gas, true_gas).sum()
            union = np.logical_or(pred_gas, true_gas).sum()
            dice_values.append((2.0 * intersection + 1e-6) / (pred_gas.sum() + true_gas.sum() + 1e-6))
            iou_values.append((intersection + 1e-6) / (union + 1e-6))

    model.train()
    return {
        "loss": total_loss / max(total, 1),
        "pixel_acc": pixel_correct / max(pixel_count, 1),
        "dice_gas": float(np.mean(dice_values)) if dice_values else 0.0,
        "iou_gas": float(np.mean(iou_values)) if iou_values else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=300)
    parser.add_argument("--test-limit", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    train_names = splits["train"][: args.limit]
    val_names = splits["val"][: args.val_limit]
    test_names = splits["test"][: args.test_limit]
    if args.smoke:
        train_names = train_names[:32]
        val_names = val_names[:16]
        test_names = test_names[:16]
        args.epochs = 1

    train_loader = DataLoader(GasLeakDataset(train_names, True), batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(GasLeakDataset(val_names, False), batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(GasLeakDataset(test_names, False), batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    if PRETRAINED_V1.exists():
        state = torch.load(PRETRAINED_V1, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"initialized from {PRETRAINED_V1.name}")
    model.to(device)
    model.train()

    class_weight = torch.tensor([0.15, 1.0, 2.5], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = []
    best_score = -1.0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.freeze_encoder_epochs:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
        else:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = True

        running_loss = 0.0
        batches = 0
        epoch_start = time.time()
        for step, (images, targets) in enumerate(train_loader, 1):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            ce = F.cross_entropy(logits, targets, weight=class_weight)
            probs = torch.softmax(logits, dim=1)
            gas_prob = probs[:, 1] + probs[:, 2]
            gas_true = (targets > 0).float()
            intersection = (gas_prob * gas_true).sum()
            dice = 1.0 - (2.0 * intersection + 1.0) / (gas_prob.sum() + gas_true.sum() + 1.0)
            loss = ce + 0.5 * dice
            loss.backward()
            optimizer.step()
            running_loss += float(loss)
            batches += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={running_loss / batches:.4f}", flush=True)

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / max(batches, 1)
        metrics["seconds"] = round(time.time() - epoch_start, 1)
        history.append(metrics)
        print(f"VALIDATION epoch={epoch} {json.dumps(metrics)}", flush=True)

        score = metrics["dice_gas"]
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), FINETUNED)
            print(f"saved best checkpoint dice={score:.4f}", flush=True)

    test_metrics = evaluate(model, test_loader, device)
    summary = {
        "history": history,
        "test": test_metrics,
        "best_val_dice": best_score,
        "train_images": len(train_names),
        "val_images": len(val_names),
        "test_images": len(test_names),
        "epochs": args.epochs,
        "image_size": [IMG_W, IMG_H],
        "total_seconds": round(time.time() - started, 1),
    }
    METRICS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TEST", json.dumps(test_metrics), flush=True)
    print(f"done in {summary['total_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
