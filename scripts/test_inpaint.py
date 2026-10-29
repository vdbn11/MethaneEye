import sys, json, numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim")
from src.config import ANNOTATIONS, IMAGES_DIR
from src.model import GasDetector
m=GasDetector()
data=json.loads(ANNOTATIONS.read_text(encoding="utf-8")); info=data["images"][0]
mask=Image.new("L",(info["width"],info["height"]),0); d=ImageDraw.Draw(mask)
for a in data["annotations"]:
    if a["image_id"]==info["id"]:
        for poly in a["segmentation"]:
            d.polygon([(poly[i],poly[i+1]) for i in range(0,len(poly)-1,2)],fill=255)
mask=mask.filter(ImageFilter.MaxFilter(15))
img=Image.open(IMAGES_DIR/info["file_name"]).convert("RGB")
blur=img.filter(ImageFilter.GaussianBlur(30))
bg=Image.composite(blur,img,mask)
# add mild noise
arr=np.array(bg).astype(np.int16)+np.random.default_rng(2).normal(0,3,np.array(bg).shape).astype(np.int16)
bg=Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
bg.save(r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim\reports\clear_inpaint.png")
r=m.predict(bg); print("inpaint",r["ratio"],r["confidence"])
print("orig",m.predict(img)["ratio"])
