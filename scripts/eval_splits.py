import json, sys, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim")
from src.config import *
from src import train as T
import segmentation_models_pytorch as smp
splits=json.loads(SPLITS.read_text(encoding="utf-8"))
m=smp.Unet(encoder_name="resnet34",encoder_weights=None,in_channels=3,classes=3)
m.load_state_dict(torch.load(PRETRAINED_V1,map_location="cpu",weights_only=True)); m.eval()
for label,names in [("first40",splits["train"][:40]),("val40",splits["val"][:40]),("test40",splits["test"][:40])]:
    loader=DataLoader(T.GasLeakDataset(names),batch_size=8)
    print(label, T.evaluate(m,loader,torch.device("cpu")))
