import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import random

# Load pre-trained model
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # Change the last layer to have 5 output classes
model.load_state_dict(torch.load('fruit_classifier_resnet2.pth', map_location=torch.device('cpu')))
model.eval()

# Image processing function
def process_image(image_path):
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    return img

# Define class labels
class_names = ['apple', 'banana', 'mango', 'orange', 'strawberry']

# Function to predict image
def predict_image(image_path):
    img = process_image(image_path)
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Function to predict and display image with predicted label
def predict_and_display(image_path):
    predicted_label = predict_image(image_path)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Tên: {class_names[predicted_label]}')
    plt.show()

# Path to the directory containing images
images_dir = 'datasets/images'

# Get list of image files in the directory
image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

# Shuffle the list of image files
random.shuffle(image_files)

# Loop through all image files, predict, and display
for image_file in image_files:
    predict_and_display(image_file)
