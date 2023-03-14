import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from PIL import ImageOps, ImageFilter
from img_to_feature_vector import *

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the CIFAR10 dataset
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# Define the function to extract features
def extract_features(img):
    # Convert the image to grayscale
    # Compute the brightness
    brightness = transforms.functional.rgb_to_grayscale(img).mean()
    # Compute the contrast
    contrast = transforms.functional.adjust_contrast(img, 2).mean() - transforms.functional.adjust_contrast(img, 0.5).mean()
    # Compute the sharpness
    sharpness = transforms.functional.adjust_sharpness(img, 2).mean() - transforms.functional.adjust_sharpness(img, 0.5).mean()
    return brightness, contrast, sharpness

# Loop through the dataset to extract features for each image
for i in range(len(dataset)):
    img, label = dataset[i]
    brightness, contrast, sharpness = extract_features(img)
    img2vec = ImgToFeatureVector(model="resnet50")
    vec = img2vec.get_vec(img)

