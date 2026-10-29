import sys, numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim")
from src.model import GasDetector
m=GasDetector(); rng=np.random.default_rng(5); w,h=480,296
coarse=rng.normal(0,1,(h//12,w//12)); coarse=Image.fromarray(((coarse-coarse.min())/(coarse.max()-coarse.min())*255).astype(np.uint8)).resize((w,h),Image.BICUBIC).filter(ImageFilter.GaussianBlur(5))
texture=np.array(coarse).astype(np.float32)
yy=np.linspace(25,95,h)[:,None]; xx=np.linspace(0,25,w)[None,:]
base=np.clip(yy+xx*0.4+texture*0.35+25,0,255).astype(np.uint8)
img=Image.fromarray(base).convert("RGB"); d=ImageDraw.Draw(img)
d.line([(0,210),(w,180)],fill=(20,20,20),width=7)
d.line([(120,296),(160,120)],fill=(28,28,28),width=5)
d.rectangle([350,60,410,100],fill=(150,150,150))
d.ellipse([60,60,95,85],fill=(120,120,120))
img=img.filter(ImageFilter.GaussianBlur(1.2))
img.save(r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim\reports\synthetic_clear.png")
r=m.predict(img); print("synthetic textured",r["ratio"],r["confidence"])
