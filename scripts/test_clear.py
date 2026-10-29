import sys
sys.path.insert(0, r"C:\Users\vdbn1\OneDrive\Documents\Default Project\methane_drone_sim")
from src.model import GasDetector, FrameLibrary
m = GasDetector()
frames = FrameLibrary()
print("clear count", len(frames.clear))
for i in range(min(28, len(frames.clear))):
    image = frames.clear_frame(i)
    result = m.predict(image)
    print(i, frames.clear[i].name, round(result["ratio"], 5), round(result["confidence"], 3))
