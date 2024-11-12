import torch
from torch.optim import Adam
import clip
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
import lpips

# Define transformations for input images
transform = transforms.Compose([
    transforms.ToTensor()  # Converts image to tensor in the range [0, 1]
])

# Load and convert the images
cat_image = Image.open("cat.webp").convert("RGB")
cat_tensor = transform(cat_image).unsqueeze(0)  # Add batch dimension

dog_image = Image.open("dog.webp").convert("RGB")
dog_tensor = transform(dog_image).unsqueeze(0)  # Add batch dimension

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

lpips_fn = lpips.LPIPS(net='vgg').to(device)  # 'vgg' is one option; 'alex' and 'squeeze' are other options

# Define the feature extractor function
def F(image_tensor):
    # Convert tensor to PIL image and then preprocess using CLIP's preprocessing
    # Ensure the image is batched
    if image_tensor.dim() == 4:
        image = transforms.ToPILImage()(image_tensor.squeeze().cpu()).convert("RGB")
    else:
        image = transforms.ToPILImage()(image_tensor.cpu()).convert("RGB")
    
    image = preprocess(image).unsqueeze(0).to(device)
    
    features = model.encode_image(image)
    return features

# Optimizer function
def optimize_poison_image(xt, xa, F, p, max_iters=1000, alpha=0.1, lr=0.01):
    # Initialize perturbation delta
    delta = torch.zeros_like(xt, requires_grad=True).to(device)

    # Define optimizer
    optimizer = Adam([delta], lr=lr)

    for _ in range(max_iters):
        optimizer.zero_grad()

        # Calculate the perturbed image
        xt_perturbed = (xt + delta).clamp(0, 1)  # Ensure values remain valid (0-1 range)

        # Extract features
        features_xt_perturbed = F(xt_perturbed)
        features_xa = F(xa)

        # Compute distance loss
        distance_loss = torch.norm(features_xt_perturbed - features_xa, p=2)

        # Regularize perturbation magnitude
        perturbation_magnitude = torch.norm(delta.view(-1), p=2)
        lpips_penalty = max(perturbation_magnitude - p, 0)

        # Total loss
        loss = distance_loss + alpha * lpips_penalty

        # Backpropagate and update
        loss.backward()
        optimizer.step()

        # Clip delta to respect the perturbation budget
        with torch.no_grad():
            delta.clamp_(-p, p)

    # Return final perturbed image
    return (xt + delta).clamp(0, 1)

# Test the function
p = 5  # Set a perturbation budget
xt = cat_tensor.to(device)  # Original image tensor (cat)
xa = dog_tensor.to(device)  # Anchor image tensor (dog)

poisoned_image = optimize_poison_image(xt, xa, F, p)
print("Poisoned image tensor:", poisoned_image)

# Remove the batch dimension if it exists and clamp values to be safe
poisoned_image = poisoned_image.squeeze().clamp(0, 1)

# Convert the tensor to a PIL image
to_pil = ToPILImage()
poisoned_image_pil = to_pil(poisoned_image.cpu())

# Display the image (optional, works in Jupyter Notebooks and some environments)
poisoned_image_pil.show()

# Save the image (optional)
poisoned_image_pil.save("poisoned_image.png")