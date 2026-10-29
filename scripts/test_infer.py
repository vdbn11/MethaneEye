import torch, numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms

model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
model.load_state_dict(torch.load(r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim\models\pretrained_unet.pth", map_location="cpu"))
model.eval()

tf = transforms.Compose([transforms.Resize((256,384)), transforms.ToTensor(), transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])

for name in ["image0001.png","image3000.png","image5000.png"]:
    img = Image.open(r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim\data\ir_images\\"+name).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]
        mask = torch.argmax(out, dim=1)[0].numpy()
    gas = (mask>0)
    print(name, img.size, "gas_ratio", float(gas.mean()), "maxprob", float(prob.max()), "classes", np.bincount(mask.flatten(), minlength=3).tolist())
    m = (mask>0).astype(np.uint8)*255
    Image.fromarray(m).resize(img.size).save(r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim\reports\\"+name.replace(".png","_mask.png"))
