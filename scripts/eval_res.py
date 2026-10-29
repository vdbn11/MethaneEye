import json, sys, numpy as np, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
sys.path.insert(0, r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim")
from src.config import *
from src.train import GasLeakDataset, evaluate
import segmentation_models_pytorch as smp
splits=json.loads(SPLITS.read_text(encoding="utf-8"))
loader=DataLoader(GasLeakDataset(splits["val"][:40]), batch_size=8)
for ck in [PRETRAINED_V1, PRETRAINED_FINAL]:
    m=smp.Unet(encoder_name="resnet34",encoder_weights=None,in_channels=3,classes=3)
    m.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True)); m.eval()
    print(ck.name, evaluate(m, loader, torch.device("cpu")))
