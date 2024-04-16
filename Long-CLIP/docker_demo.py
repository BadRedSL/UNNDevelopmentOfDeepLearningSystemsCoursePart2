from model import longclip
from PIL import Image
import torch
import numpy as np
import datetime
import warnings

warnings.filterwarnings("ignore")

start = datetime.datetime.now()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)

prompts = list(np.loadtxt(f"./app_volume/text/test_text.txt", dtype=str, quotechar='"'))
image_path = f"./app_volume/img/test_image.jpg"
finish = datetime.datetime.now()
print(f"Preparing model time: {(finish - start).total_seconds()} sec\n")

start = datetime.datetime.now()
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = longclip.tokenize(prompts).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
print(f"Best prompt: {prompts[[i for i, v in enumerate(probs[0]) if v == max(probs[0])][0]]}\n")
finish = datetime.datetime.now()
print(f"Process time: {(finish - start).total_seconds()} sec")
