import os
from dotenv import load_dotenv
import numpy as np
from collections import Counter
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import StratifiedShuffleSplit
import json5
from model_cnn import SimpleCNN

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    learning_rate = 0.002
    epochs = 50

    load_dotenv()
    config_used = "layer_config"
    categories_dir = os.getenv("CATEGORIES_DIR", "categories")
    model_config_path = os.getenv("CONFIG_PATH", "model_small.jsonc")
    # Load model config
    with open(model_config_path, "r") as f:
        model_config = json5.load(f)
    image_size = model_config["imageSize"]
    layer_config = model_config[config_used]
    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=categories_dir, transform=transform)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # Build targets array, skipping unreadable images
    valid_indices = []
    targets = []
    for idx in range(len(dataset.imgs)):
        path, label = dataset.imgs[idx]
        try:
            img = Image.open(path)
            img.verify()  # Check if image is valid
            valid_indices.append(idx)
            targets.append(label)
        except Exception:
            print(f"Skipping corrupted image: {path}")
    targets = np.array(targets)
    class_names = dataset.classes
    print("Class distribution before clipping:")
    label_counts = Counter(targets)
    for label, count in sorted(label_counts.items()):
        print(f"  {class_names[label]}: {count} images")

    # Clip each class to the median count
    from collections import defaultdict
    import random
    random.seed(42)
    indices_by_class = defaultdict(list)
    for idx, label in zip(valid_indices, targets):
        indices_by_class[label].append(idx)
    counts = [len(indices_by_class[label]) for label in range(len(class_names))]
    median_count = int(np.median(counts))
    print(f"Clipping each class to median count: {median_count}")
    clipped_indices = []
    for label, idxs in indices_by_class.items():
        if len(idxs) > median_count:
            clipped = random.sample(idxs, median_count)
        else:
            clipped = idxs
        clipped_indices.extend(clipped)
    # Rebuild targets for clipped set
    clipped_targets = [dataset.imgs[i][1] for i in clipped_indices]
    print("Class distribution after clipping:")
    clipped_label_counts = Counter(clipped_targets)
    for label, count in sorted(clipped_label_counts.items()):
        print(f"  {class_names[label]}: {count} images")
    # Rebuild targets for clipped set
    clipped_targets = [dataset.imgs[i][1] for i in clipped_indices]
    print("Class distribution after clipping:")
    clipped_label_counts = Counter(clipped_targets)
    for label, count in sorted(clipped_label_counts.items()):
        print(f"  {class_names[label]}: {count} images")

    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(clipped_targets)), clipped_targets))
    train_dataset = Subset(dataset, [clipped_indices[i] for i in train_idx])
    test_dataset = Subset(dataset, [clipped_indices[i] for i in test_idx])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = get_device()
    model = SimpleCNN(layer_config, num_classes=len(class_names), input_size=(3, image_size, image_size)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    def train_epoch(model, loader, optimizer, criterion, device, epoch):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if batch_idx % 100 == 0 and batch_idx != 0:
                print(f'Epoch {epoch:03d}, Batch {batch_idx:03d}, Loss: {loss.item():.6f}')
        train_loss = running_loss / len(loader)
        train_accuracy = correct / total if total > 0 else 0
        return train_loss, train_accuracy
    def test_epoch(model, loader, criterion, device):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        test_loss = test_loss / len(loader)
        test_accuracy = correct / total if total > 0 else 0
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_loss, test_accuracy = test_epoch(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    # Save model
    model_path = os.getenv("MODEL_PATH", "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
if __name__ == "__main__":
    main()
