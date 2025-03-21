from model import ViTConXWithAvgPooling
import torch

def load_model(device):
    model_path = "best_model.pth"  # Ensure the correct path

    model = ViTConXWithAvgPooling(num_classes=1).to(device)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model

