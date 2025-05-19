import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from model import load_model
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)

class_names = [
    'Alzheimer Disease', 
    'Mild Alzheimer Risk', 
    'Moderate Alzheimer Risk',
    'Very Mild Alzheimer Risk', 
    'No Risk', 
    'Parkinson Disease'
]

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted = torch.argmax(probs).item()
        confidence = probs[predicted].item() * 100
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Vbai-DPA 2.2 (C Version)",
    description="Upload an MRI and fMRI image to classify the risk level using the 'C' version of the Vbai-DPA 2.2 model."
).launch()
