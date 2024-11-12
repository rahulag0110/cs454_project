from PIL import Image
import torch
from torchvision import transforms

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor (0-255 -> 0-1 range)
])

# Load and convert the cat.webp image
cat_image = Image.open("cat.webp").convert("RGB")  # Convert to RGB to ensure 3 channels
cat_tensor = transform(cat_image)  # Apply the transformations

# Load and convert the dog.webp image
dog_image = Image.open("dog.webp").convert("RGB")  # Convert to RGB to ensure 3 channels
dog_tensor = transform(dog_image)  # Apply the transformations

# Print the shape of the resulting tensors
print("Cat Tensor Shape:", cat_tensor.shape)
print("Dog Tensor Shape:", dog_tensor.shape)
