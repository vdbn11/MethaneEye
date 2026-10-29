import json, torch, numpy as np
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp
from torchvision import transforms
from pathlib import Path

root = Path(r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim")
ann = json.loads((root/"data/annotations.json").read_text(encoding="utf-8"))
by_img = {}
for a in ann["annotations"]: by_img.setdefault(a["image_id"], []).append(a)
tf = transforms.Compose([transforms.Resize((320,480)), transforms.ToTensor(), transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])

def gt_mask(info, anns):
    m = Image.new("L",(info["width"],info["height"]),0); d=ImageDraw.Draw(m)
    for a in anns:
        for poly in a["segmentation"]:
            d.polygon([(poly[i],poly[i+1]) for i in range(0,len(poly)-1,2)], fill=255)
    return np.array(m)>0

for ck in ["pretrained_unet.pth","../Temp/opencode/skysniff/unet_gas_leak_segmentation.pth"]:
    path = (root/"models"/ck) if "pretrained" in ck else Path(r"C:\Users\vdbn1\AppData\Local\Temp\opencode\skysniff\unet_gas_leak_segmentation.pth")
    print("CHECKPOINT", path)
    model=smp.Unet(encoder_name="resnet34",encoder_weights=None,in_channels=3,classes=3)
    try:
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)); model.eval()
    except Exception as e:
        print("load err",e); continue
    for info in ann["images"][:3]:
        anns=by_img.get(info["id"],[])
        if not anns: continue
        img=Image.open(root/"data/ir_images"/info["file_name"]).convert("RGB")
        with torch.no_grad():
            out=model(tf(img).unsqueeze(0)); p=torch.softmax(out,1)[0]
        gt=gt_mask(info,anns)
        for th in [0.05,0.1,0.2,0.5]:
            pred=(p[1]+p[2]>th).numpy()
            pred=np.array(Image.fromarray((pred*255).astype(np.uint8)).resize((info["width"],info["height"]),Image.NEAREST))>0
            inter=(pred&gt).sum(); union=(pred|gt).sum(); iou=inter/union if union else 1
            print(info["file_name"],"th",th,"iou=%.3f pred=%.4f gt=%.4f"%(iou,pred.mean(),gt.mean()))
