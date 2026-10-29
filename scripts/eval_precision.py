import json,sys,numpy as np,torch
from torch.utils.data import DataLoader
sys.path.insert(0, r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim")
from src.config import *
from src.train import GasLeakDataset
import segmentation_models_pytorch as smp
splits=json.loads(SPLITS.read_text(encoding="utf-8")); names=splits["val"][:100]
loader=DataLoader(GasLeakDataset(names),batch_size=8)
m=smp.Unet(encoder_name="resnet34",encoder_weights=None,in_channels=3,classes=3)
m.load_state_dict(torch.load(FINETUNED,map_location="cpu",weights_only=True)); m.eval()
tp=fp=fn=tn=0; gt_ratio=[]; pr_ratio=[]
with torch.no_grad():
 for x,y in loader:
  p=torch.argmax(m(x),1)
  pg=(p>0).numpy(); tg=(y>0).numpy()
  tp+=np.logical_and(pg,tg).sum(); fp+=np.logical_and(pg,~tg).sum(); fn+=np.logical_and(~pg,tg).sum(); tn+=np.logical_and(~pg,~tg).sum()
  pr_ratio.append(pg.mean()); gt_ratio.append(tg.mean())
prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); iou=tp/(tp+fp+fn+1e-9)
print("precision %.3f recall %.3f iou %.3f pred_ratio %.3f gt_ratio %.3f"%(prec,rec,iou,np.mean(pr_ratio),np.mean(gt_ratio)))
