from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "ir_images"
MASKS_DIR = DATA_DIR / "masks"
ANNOTATIONS = DATA_DIR / "annotations.json"
SPLITS = DATA_DIR / "splits.json"
CARBON_MAPPER_CSV = DATA_DIR / "carbon_mapper_sau.csv"
MODELS_DIR = ROOT / "models"
PRETRAINED_V1 = MODELS_DIR / "pretrained_unet_v1.pth"
PRETRAINED_FINAL = MODELS_DIR / "pretrained_unet.pth"
FINETUNED = MODELS_DIR / "methane_unet_finetuned.pth"
METRICS = MODELS_DIR / "metrics.json"
WEB_DIR = ROOT / "web"
REPORTS_DIR = ROOT / "reports"

IMG_W = 384
IMG_H = 256
NUM_CLASSES = 3
CLASS_NAMES = ["background", "gas_day", "gas_night"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787

SITE_LAT = 25.9587
SITE_LON = 49.6775
GAS_RATIO_THRESHOLD = 0.008
GAS_CONFIDENCE_THRESHOLD = 0.58
