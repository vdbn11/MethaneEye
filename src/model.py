import base64
import io
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torchvision.transforms import functional as TF

from src.config import (
    ANNOTATIONS,
    FINETUNED,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_H,
    IMG_W,
    IMAGES_DIR,
    METRICS,
    NUM_CLASSES,
    PRETRAINED_FINAL,
    PRETRAINED_V1,
)

import segmentation_models_pytorch as smp


class GasDetector:
    def __init__(self, checkpoint: Optional[Path] = None):
        if checkpoint is None:
            checkpoint = FINETUNED if FINETUNED.exists() else PRETRAINED_V1
        self.checkpoint = checkpoint
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        torch.set_num_threads(max(4, min(8, torch.get_num_threads())))
        self.metrics = {}
        if METRICS.exists():
            try:
                self.metrics = json.loads(METRICS.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    def predict(self, image: Image.Image) -> dict:
        original = image.convert("RGB")
        width, height = original.size
        tensor = TF.to_tensor(TF.resize(original, [IMG_H, IMG_W]))
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)

        started = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            mask = torch.argmax(logits, dim=1)[0].numpy().astype(np.uint8)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        gas_probability = (probabilities[1] + probabilities[2]).numpy()
        gas_mask = mask > 0
        ratio = float(gas_mask.mean())
        confidence = float(gas_probability[gas_mask].mean()) if gas_mask.any() else float(gas_probability.max())
        centroid = (width / 2.0, height / 2.0)
        box = None
        if gas_mask.any():
            ys, xs = np.where(gas_mask)
            centroid = (float(xs.mean()) / IMG_W * width, float(ys.mean()) / IMG_H * height)
            x0, x1 = int(xs.min() / IMG_W * width), int(xs.max() / IMG_W * width)
            y0, y1 = int(ys.min() / IMG_H * height), int(ys.max() / IMG_H * height)
            box = [x0, y0, x1, y1]

        full_mask = np.array(
            Image.fromarray(gas_mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
        ) > 0
        return {
            "mask": full_mask,
            "ratio": ratio,
            "confidence": confidence,
            "centroid": centroid,
            "box": box,
            "inference_ms": elapsed_ms,
            "model": self.checkpoint.name,
        }

    def annotate(self, image: Image.Image, result: dict, telemetry: dict) -> Image.Image:
        base = image.convert("RGB")
        width, height = base.size
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if result.get("mask") is not None and result["mask"].any():
            color_layer = Image.new("RGBA", base.size, (255, 64, 24, 0))
            alpha = Image.fromarray((result["mask"].astype(np.uint8) * 120))
            color_layer.putalpha(alpha)
            overlay = Image.alpha_composite(overlay, color_layer)
            draw = ImageDraw.Draw(overlay)

        if result.get("box"):
            x0, y0, x1, y1 = result["box"]
            pad = 8
            draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=(255, 196, 0, 255), width=2)
            draw.text((x0 + 2, max(2, y0 - 22)), "METHANE PLUME", fill=(255, 196, 0, 255), font=_font(14))

        draw.line([(width / 2 - 18, height / 2), (width / 2 + 18, height / 2)], fill=(0, 255, 210, 190), width=1)
        draw.line([(width / 2, height / 2 - 18), (width / 2, height / 2 + 18)], fill=(0, 255, 210, 190), width=1)
        draw.ellipse([width / 2 - 22, height / 2 - 22, width / 2 + 22, height / 2 + 22], outline=(0, 255, 210, 90), width=1)

        draw.rectangle([0, 0, width, 30], fill=(4, 12, 16, 190))
        draw.rectangle([0, height - 58, width, height], fill=(4, 12, 16, 200))
        draw.text((10, 7), "OGI / MWIR SIMULATED FEED", fill=(0, 255, 210), font=_font(14))
        draw.text(
            (width - 150, 7),
            telemetry.get("mode", "AUTO"),
            fill=(255, 196, 0) if result["ratio"] > 0.001 else (120, 220, 160),
            font=_font(14),
        )
        draw.text(
            (10, height - 54),
            f"GPS {telemetry.get('lat', 0):.6f}, {telemetry.get('lon', 0):.6f}   ALT {telemetry.get('alt', 0):.0f}m   HDG {telemetry.get('heading', 0):03.0f}   BAT {telemetry.get('battery', 0):.0f}%",
            fill=(210, 232, 236),
            font=_font(11),
        )
        draw.text(
            (10, height - 32),
            f"GAS {result['ratio'] * 100:5.2f}%   CONF {result['confidence'] * 100:5.1f}%",
            fill=(255, 120, 90) if result["ratio"] > 0.001 else (120, 220, 160),
            font=_font(11),
        )

        return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def _font(size: int):
    for candidate in ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


class FrameLibrary:
    def __init__(self):
        data = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
        annotated = {ann["image_id"] for ann in data["annotations"]}
        gas = []
        clear = []
        for info in sorted(data["images"], key=lambda item: item["id"]):
            path = IMAGES_DIR / info["file_name"]
            if not path.exists():
                continue
            if info["id"] in annotated:
                if info["width"] >= 450 and info["id"] < 500:
                    gas.append(path)
            else:
                clear.append(path)
        self.gas = gas or [IMAGES_DIR / "image0001.png"]
        self.clear = clear
        self._synthetic_clear = None

    def gas_frame(self, index: int) -> Image.Image:
        path = self.gas[index % len(self.gas)]
        return Image.open(path).convert("RGB")

    def clear_frame(self, index: int) -> Image.Image:
        if self._synthetic_clear is None:
            width, height = 480, 296
            rng = np.random.default_rng(5)
            coarse = rng.normal(0, 1, (height // 12, width // 12))
            coarse = Image.fromarray(
                ((coarse - coarse.min()) / (coarse.max() - coarse.min()) * 255).astype(np.uint8)
            ).resize((width, height), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(5))
            texture = np.array(coarse).astype(np.float32)
            yy = np.linspace(25, 95, height)[:, None]
            xx = np.linspace(0, 25, width)[None, :]
            base = np.clip(yy + xx * 0.4 + texture * 0.35 + 25, 0, 255).astype(np.uint8)
            image = Image.fromarray(base).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.line([(0, 210), (width, 180)], fill=(20, 20, 20), width=7)
            draw.line([(120, height), (160, 120)], fill=(28, 28, 28), width=5)
            draw.rectangle([350, 60, 410, 100], fill=(150, 150, 150))
            draw.ellipse([60, 60, 95, 85], fill=(120, 120, 120))
            draw.line([(0, 245), (width, 235)], fill=(38, 38, 38), width=3)
            self._synthetic_clear = image.filter(ImageFilter.GaussianBlur(1.2))
        return self._synthetic_clear.copy()


def encode_jpeg(image: Image.Image, quality: int = 82) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
