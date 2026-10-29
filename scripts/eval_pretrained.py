import json, torch, numpy as np
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp
from torchvision import transforms
from pathlib import Path

root = Path(r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim")
ann = json.loads((root/"data/annotations.json").read_text(encoding="utf-8"))
by_img = {}
for a in ann["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)

model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
model.load_state_dict(torch.load(root/"models/pretrained_unet.pth", map_location="cpu", weights_only=True))
model.eval()
tf = transforms.Compose([transforms.Resize((320,480)), transforms.ToTensor(), transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])

def gt_mask(info, anns):
    m = Image.new("L", (info["width"], info["height"]), 0)
    d = ImageDraw.Draw(m)
    for a in anns:
        seg = a["segmentation"]
        if isinstance(seg, list):
            for poly in seg:
                pts = [(poly[i], poly[i+1]) for i in range(0, len(poly)-1, 2)]
                d.polygon(pts, fill=255)
    return np.array(m) > 0

ious=[]; dices=[]; ratios=[]
for info in ann["images"][:12]:
    anns = by_img.get(info["id"], [])
    if not anns: continue
    img = Image.open(root/"data/ir_images"/info["file_name"]).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        pred = (torch.argmax(out,1)[0].numpy()>0)
    pred = np.array(Image.fromarray(pred.astype(np.uint8)*255).resize((info["width"],info["height"]), Image.NEAREST))>0
    gt = gt_mask(info, anns)
    inter=(pred&gt).sum(); union=(pred|gt).sum()
    iou=inter/union if union else 1; dice=2*inter/(pred.sum()+gt.sum())
    ious.append(iou); dices.append(dice); ratios.append((pred.mean(),gt.mean()))
    print(info["file_name"], info["width"], info["height"], "iou=%.3f dice=%.3f pred=%.4f gt=%.4f"%(iou,dice,pred.mean(),gt.mean()))
print("MEAN iou=%.3f dice=%.3f"%(np.mean(ious),np.mean(dices)))
