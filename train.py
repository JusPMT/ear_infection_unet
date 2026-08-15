import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import load_dataset
from models.unet_2015.unet_model import UNet
import os

def train_model(data_dir, epochs, batch_size, learning_rate, img_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print(f"Loading data from {data_dir}...")
    train_loader, val_loader, class_names = load_dataset(data_dir, img_size=img_size, batch_size=batch_size)
    n_classes = len(class_names)
    print(f"Found {n_classes} classes: {class_names}")

    # 2. Initialize Model (Modified U-Net)
    # n_channels=3 for RGB. n_classes is our number of output classes.
    model = UNet(n_channels=3, n_classes=n_classes, bilinear=True)
    model.to(device)

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    best_val_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_unet_classifier.pth')
            print(f"--> Saved best model with accuracy {best_val_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Modified U-Net for Image Classification")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--img_size', type=int, default=256, help="Image size for resizing")
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr, args.img_size)
