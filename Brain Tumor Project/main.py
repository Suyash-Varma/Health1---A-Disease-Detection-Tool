import gradio as gr
import requests
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# model = torch.load('Tumordensenet201.pt')
# model.eval()

model = torch.load('Tumordensenet201.pt')
device = torch.device('cpu')
model.to(device)
# Set the model to evaluation mode
model.eval()

labels = ['Meningioma', 'Glioma', 'Pituitary tumor']
def predict(inp):
  transformed_img = transform(inp)
#   print(transformed_img.shape)
  transformed_img_4d = transformed_img.unsqueeze(0)
#   print(transformed_img_4d.shape)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(transformed_img_4d)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(3)}
  return confidences

css_code='body{background-image:url("https://assets.technologynetworks.com/production/dynamic/images/content/354432/early-detection-of-brain-tumors-and-beyond-354432-960x540.jpg?cb=11900964");}'

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3),
             examples=["2310.png", "7.png","931.png"],title='Brain tumor Classification',css=css_code).launch()

