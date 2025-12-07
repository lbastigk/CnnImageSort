import os
from dotenv import load_dotenv
import shutil
import torch
import json5
from PIL import Image
from torchvision import transforms
from model_cnn import SimpleCNN

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model(model_path, config_path):
    with open(config_path, "r") as f:
        model_config = json5.load(f)
    image_size = model_config["imageSize"]
    layer_config = model_config["layer_config"]
    # Load class names from categories
    categories_dir = os.getenv("CATEGORIES_DIR", "categories")
    class_names = sorted([d for d in os.listdir(categories_dir) if os.path.isdir(os.path.join(categories_dir, d))])
    model = SimpleCNN(layer_config, num_classes=len(class_names), input_size=(3, image_size, image_size))
    model.load_state_dict(torch.load(model_path, map_location=get_device()))
    model.eval()
    return model, class_names, image_size

def sort_images():
    load_dotenv()
    model_path = os.getenv("MODEL_PATH", "trained_model.pth")
    config_path = os.getenv("CONFIG_PATH", "model_small.jsonc")
    tosort_dir = os.getenv("TOSORT_DIR", "tosort")
    suggestions_dir = os.getenv("SUGGESTIONS_DIR", "suggestions")
    model, class_names, image_size = load_model(model_path, config_path)
    device = get_device()
    model = model.to(device)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    os.makedirs(suggestions_dir, exist_ok=True)
    for cat in class_names:
        os.makedirs(os.path.join(suggestions_dir, cat), exist_ok=True)
    for fname in os.listdir(tosort_dir):
        fpath = os.path.join(tosort_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            pred_cat = class_names[pred]
        dest_dir = os.path.join(suggestions_dir, pred_cat)
        dest_path = os.path.join(dest_dir, fname)
        shutil.move(fpath, dest_path)
        print(f"Sorted {fname} -> {pred_cat}")
if __name__ == "__main__":
    sort_images()
