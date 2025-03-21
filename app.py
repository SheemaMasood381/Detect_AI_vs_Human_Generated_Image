import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from load_model import load_model

# Load the trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_model(DEVICE)
MODEL.eval()

# Define image preprocessing (same as training)
inference_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    """Runs inference on an image and returns the prediction."""
    image = Image.open(image).convert("RGB")
    image = inference_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = MODEL(image).squeeze()  # Ensure correct dimensions
        print("Raw model output:", output)
        
        probability = torch.sigmoid(output).cpu().numpy().item()
        print("Sigmoid probability:", probability)
    
    result = "The image is **AI-Generated** 🖥️" if probability > 0.5 else "The image is **Human-Generated** 🎨"
    return result
    


# Define the Gradio interface
demo = gr.Blocks()
with demo:
    gr.Markdown("# 🔍 AI vs Human Image Detector")
    gr.Markdown("Upload an image to check if it's **AI-generated** or **human-made**.")
    
    with gr.Row():
        image_input = gr.Image(type="filepath")
    
    with gr.Row():
        output_text = gr.Markdown("")
    
    submit_button = gr.Button("Analyze Image")
    submit_button.click(predict, inputs=image_input, outputs=output_text)
    
    # Footer Section
    gr.Markdown("<hr>")  # Adds a line separator
    gr.Markdown("<p style='text-align: center; font-size: 14px;'>🌟 Developed by <b>Sheema Masood</b> | Built with <b>Gradio</b> 🌟</p>")
    
# Run the Gradio app
demo.launch()

