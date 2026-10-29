import sys, torch
from PIL import Image
sys.path.insert(0, r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim")
from src.model import GasDetector
from src.config import IMAGES_DIR
m=GasDetector()
for i in range(1,12):
    r=m.predict(Image.open(IMAGES_DIR/f"image{i:04d}.png").convert("RGB"))
    print(i, round(r["ratio"],4), round(r["confidence"],3), r["inference_ms"])
