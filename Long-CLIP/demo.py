from model import longclip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)

prompts = ["A man is crossing the street with a red car parked nearby.",
           "A man is driving a car in an urban scene.",
           "A man crossed the river on an ice bridge, with a modern city in the background.",
           "A man crossed the lava lake on an snow bridge, with a modern city in the background.",
           "A dog crossed the river on an ice bridge, with a modern city in the background."]
text = longclip.tokenize(prompts).to(device)
image = preprocess(Image.open("img/test_image_1.jpg")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
print(f"Best prompt: {prompts[[i for i, v in enumerate(probs[0]) if v == max(probs[0])][0]]}")
