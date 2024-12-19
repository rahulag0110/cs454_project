import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = Image.open("C:/Users/Anca/Documents/GitHub/cs454_project/code/poisoned_image.png").convert("RGB")  # Replace with your image path
image_input = preprocess(image).unsqueeze(0).to(device)  

text_labels = ["a cat", "a dog", "a car", "a tree", "a house"]

text_inputs = clip.tokenize(text_labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

similarity_scores = (image_features @ text_features.T).squeeze()  

probs = similarity_scores.softmax(dim=-1).cpu().numpy()

for i, label in enumerate(text_labels):
    print(f"Similarity score for '{label}': {similarity_scores[i].item():.4f} (probability: {probs[i]:.4f})")
