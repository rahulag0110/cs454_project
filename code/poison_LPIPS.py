import torch
from torch.optim import Adam
import clip
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
import lpips

transform = transforms.Compose([
    transforms.ToTensor()  
])

cat_image = Image.open("cat.webp").convert("RGB")
cat_tensor = transform(cat_image).unsqueeze(0)  

dog_image = Image.open("dog.webp").convert("RGB")
dog_tensor = transform(dog_image).unsqueeze(0)  


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


lpips_fn = lpips.LPIPS(net='vgg').to(device)


def F(image_tensor):

    image = transforms.ToPILImage()(image_tensor.cpu()).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    features = model.encode_image(image)

    return features

def optimize_poison_image(xt, xa, F, lpips_fn, p, max_iters=100, alpha=0.1, lr=0.1):

    delta = (torch.rand_like(xt) * 2 - 1).clamp(-p, p).requires_grad_(True).to(device)

    optimizer = Adam([delta], lr=lr)

    def normalize_for_lpips(image):
        return (image * 2) - 1 

    for iteri in range(max_iters):
        optimizer.zero_grad()
        lpips_fn.train()

        print(f"Iteation: {iteri}")

        xt_perturbed = xt + delta 

        features_xt_perturbed = F(xt_perturbed)
        features_xa = F(xa)

        distance_loss = torch.norm(features_xt_perturbed - features_xa, p=2)

        lpips_distance = lpips_fn(normalize_for_lpips(xt_perturbed), normalize_for_lpips(xt)).mean()

        lpips_penalty = torch.relu(lpips_distance - p)

        print(f"Distance Loss: {distance_loss.item()}, LPIPS Distance: {lpips_distance.item()}")
        print(f"LPIPS Penalty: {lpips_penalty.item()}")

        loss = distance_loss + alpha * lpips_penalty

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(-p, p)

    return (xt + delta).clamp(0, 1)


p = 0.07  # Visual perturbation budget
xt = cat_tensor.to(device)  
xa = dog_tensor.to(device)  

poisoned_image = optimize_poison_image(xt, xa, F, lpips_fn, p)
print("Poisoned image tensor:", poisoned_image)

poisoned_image = poisoned_image.squeeze().clamp(0, 1)

to_pil = ToPILImage()
poisoned_image_pil = to_pil(poisoned_image.cpu())



poisoned_features = F(poisoned_image.unsqueeze(0).to(device))
target_features = F(xa)
feature_distance = torch.norm(poisoned_features - target_features, p=2).item()
print(f"Feature Distance: {feature_distance}")

lpips_distance = lpips_fn(poisoned_image.unsqueeze(0).to(device), xt).item()
print(f"LPIPS Distance: {lpips_distance}")


poisoned_image_pil.show()

poisoned_image_pil.save("poisoned_image_lpips.png")
