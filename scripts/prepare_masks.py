import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ANNOTATIONS, IMAGES_DIR, MASKS_DIR, SPLITS


def polygon_mask(size, annotations):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for ann in annotations:
        category = int(ann["category_id"])
        segmentation = ann.get("segmentation") or []
        if isinstance(segmentation, dict):
            continue
        for poly in segmentation:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            points = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
            draw.polygon(points, fill=category)
    return mask


def main():
    data = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    grouped = {}
    for ann in data["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)

    images = sorted(data["images"], key=lambda item: item["id"])
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    split_names = []
    class_pixels = Counter()
    for index, info in enumerate(images):
        source = IMAGES_DIR / info["file_name"]
        if not source.exists():
            continue
        target = MASKS_DIR / (source.stem + ".png")
        if not target.exists():
            mask = polygon_mask((info["width"], info["height"]), grouped.get(info["id"], []))
            mask.save(target, optimize=True)
        split_names.append(info["file_name"])

    total = len(split_names)
    runs = []
    start = 0
    previous = (images[0]["width"], images[0]["height"])
    for index, info in enumerate(images[1:], 1):
        size = (info["width"], info["height"])
        if size != previous:
            runs.append((start, index - 1, previous))
            start = index
            previous = size
    runs.append((start, len(images) - 1, previous))

    train_groups = [0, 1, 2, 3, 6, 7, 9]
    val_groups = [4, 5]
    test_groups = [8]
    caps = {0: 350, 1: 250, 2: 200, 3: 150, 6: 200, 7: 150, 9: 300}

    def pick(group_index, cap):
        first, last, _ = runs[group_index]
        names = split_names[first : last + 1]
        if cap is None or len(names) <= cap:
            return names
        positions = np.linspace(0, len(names) - 1, cap).astype(int)
        return [names[position] for position in positions]

    train = []
    for group in train_groups:
        train.extend(pick(group, caps.get(group, 250)))
    val = []
    for group in val_groups:
        val.extend(pick(group, 250))
    test = pick(test_groups[0], 420)

    splits = {
        "train": train,
        "val": val,
        "test": test,
        "groups": [
            {"index": index, "first": first, "last": last, "size": list(size)}
            for index, (first, last, size) in enumerate(runs)
        ],
        "strategy": "grouped by source-video resolution runs; unseen videos in test",
    }
    SPLITS.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    for name in split_names[:50]:
        mask = Image.open(MASKS_DIR / (Path(name).stem + ".png"))
        class_pixels.update(mask.getdata())

    print(f"images={total} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print("groups:", [(run[2], run[1] - run[0] + 1) for run in runs])
    print("sample pixel classes:", dict(class_pixels))


if __name__ == "__main__":
    main()
