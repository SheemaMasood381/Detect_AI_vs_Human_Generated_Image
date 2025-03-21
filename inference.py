import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from load_model import load_model

# ✅ Define the image transformations (same as used during training)
inference_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path, model, device="cpu"):
    """Runs inference on a single image and returns the prediction."""
    image = Image.open(image_path).convert("RGB")
    image = inference_transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image).squeeze(1)
        probability = torch.sigmoid(output).item()
    
    return {"file_name": os.path.basename(image_path), "prediction": "AI-Generated" if probability > 0.5 else "Human-Generated"}

def predict_multiple_images(image_paths, model, device="cpu"):
    """Runs inference on multiple images and returns predictions."""
    results = []
    for img_path in image_paths:
        result = predict_image(img_path, model, device)
        results.append(result)
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    
    # ✅ Single Image Prediction
    image_path = "sample.jpg"  # Change this to your image path
    single_result = predict_image(image_path, model, device)
    print(f"Single Image Prediction: {single_result}")

    # ✅ Multiple Images Prediction
    image_folder = "test_images"  # Folder containing test images
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(("png", "jpg", "jpeg"))]

    if image_paths:
        multiple_results = predict_multiple_images(image_paths, model, device)
        for res in multiple_results:
            print(f"Multiple Image Prediction: {res}")
    else:
        print("No valid images found in the folder.")
