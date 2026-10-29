import sys, numpy as np
from PIL import Image, ImageFilter
sys.path.insert(0, r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim")
from src.model import GasDetector, FrameLibrary
m = GasDetector(); frames = FrameLibrary()
rng = np.random.default_rng(3)
w,h = 480,296
yy = np.linspace(30, 110, h)[:,None]
xx = np.linspace(0, 35, w)[None,:]
noise = rng.normal(0, 4, (h,w))
synthetic = Image.fromarray(np.clip(yy+xx+noise+30,0,255).astype(np.uint8)).convert("RGB")
variants = {"synthetic": synthetic}
base = frames.clear_frame(0)
variants["blur15"] = base.filter(ImageFilter.GaussianBlur(15))
variants["blur30"] = base.filter(ImageFilter.GaussianBlur(30))
variants["dark"] = Image.eval(base.convert("L"), lambda v: int(v*0.5)).convert("RGB")
for name,img in variants.items():
    r=m.predict(img); print(name, round(r["ratio"],5), round(r["confidence"],3), r["inference_ms"])
