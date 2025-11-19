import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
from PIL import Image
import yaml

from src.models.cnn import CNN
from src.data.datasets import get_transforms


def predict():
    config_path = os.path.join(project_root, "config", "training.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from src.data.datasets import ImageFolderDataset

    train_dir = os.path.join(project_root, config["data"]["train_dir"])
    train_dataset = ImageFolderDataset(train_dir)
    classes = train_dataset.classes
    print(f"Classes: {classes}")

    model = CNN(num_classes=len(classes))
    model_path = os.path.join(project_root, "outputs", "models", "best_model.pth")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    else:
        print(f"Model not found at {model_path}")
        return

    transform = get_transforms(config["data"]["img_size"], is_train=False)

    def predict_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return predicted.item(), confidence.item()

    pred_dir = os.path.join(project_root, config["data"]["pred_dir"])
    if os.path.exists(pred_dir):
        print(f"\nMaking predictions for images in {pred_dir}:")
        for img_name in os.listdir(pred_dir):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(pred_dir, img_name)
                class_id, confidence = predict_image(img_path)
                class_name = classes[class_id] if class_id < len(classes) else "unknown"
                print(
                    f"Image: {img_name} -> Class: {class_name} (ID: {class_id}), Confidence: {confidence:.2f}"
                )


if __name__ == "__main__":
    predict()
