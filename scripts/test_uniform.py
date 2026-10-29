import sys, numpy as np
from PIL import Image
sys.path.insert(0, r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim")
from src.model import GasDetector
m=GasDetector(); print("ckpt",m.checkpoint.name)
w,h=480,296
imgs={
 "black":Image.new("RGB",(w,h),(5,5,5)),
 "gray":Image.new("RGB",(w,h),(90,90,90)),
 "white":Image.new("RGB",(w,h),(220,220,220)),
 "gradient":Image.fromarray(np.tile(np.linspace(40,160,w,dtype=np.uint8),(h,1))).convert("RGB"),
 "noise":Image.fromarray(np.random.default_rng(1).integers(60,140,(h,w),dtype=np.uint8)).convert("RGB"),
}
for name,img in imgs.items():
    r=m.predict(img); print(name, round(r["ratio"],5), round(r["confidence"],3))
