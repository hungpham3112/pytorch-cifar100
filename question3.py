import torch
import pandas as pd
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import argparse
from models.senet import custom_seresnet34
import os
import random

# Set up argument parser
parser = argparse.ArgumentParser(description='Predict Cangjie labels for character images using a trained CNN model.')
parser.add_argument('--model', type=str, required=True, help='Path to the trained model .pth file.')
parser.add_argument('--txt_file', type=str, required=True, help='Path to the labels txt file.')
parser.add_argument('--image', type=str, required=True, help='Path to the image to be predicted.')
args = parser.parse_args()

# Load the trained model
model = custom_seresnet34()
model.load_state_dict(torch.load(args.model))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the txt file with labels
labels_df = pd.read_csv(args.txt_file, delimiter=' ', names=['label', 'character', 'JISx0208', 'UTF8', 'Cangjie'])

# Define custom transform
class InvertColors(object):
    def __call__(self, img):
        return TF.invert(img)

# Define the image transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.RandomAffine(degrees=15, scale=(0.8, 1.2), shear=10),
    transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
    InvertColors(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def transform_image(image_path, save_path):
    # Open the image
    image = Image.open(image_path)
    
    # Apply transformations
    transformed_image = transform(image)
    
    return transformed_image

def predict_character(image_path, model, labels_df, device):
    # Generate a filename for the transformed image
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    transformed_image_path = f"{name}_transformed{ext}"
    
    # Transform the image and save it
    image_tensor = transform_image(image_path, transformed_image_path)
    
    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Predict the label
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()
    
    # Get the corresponding character and Cangjie encoding
    row = labels_df.iloc[predicted_label]
    character = row['character']
    cangjie = row['Cangjie']
    
    return predicted_label, character, cangjie, transformed_image_path

# Test the prediction function
predicted_label, character, cangjie, transformed_image_path = predict_character(args.image, model, labels_df, device)

# Print the results
print(f"Predicted Label Index: {predicted_label}")
print(f"Character: {character}")
print(f"Cangjie Encoding: {cangjie}")
