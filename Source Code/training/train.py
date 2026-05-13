import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

def main():
    # -------------------------------
    # Device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------
    # Transforms
    # -------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
    ])

    # -------------------------------
    # Dataset
    # -------------------------------
    data_dir = "data/PlantVillage-Dataset/raw/color"
    print("Loading dataset...")

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    print("Total images:", len(dataset))

    # -------------------------------
    # Split dataset
    # -------------------------------
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    print("Dataset split done")

    # -------------------------------
    # DataLoaders (SAFE for Windows)
    # -------------------------------
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=0)

    # -------------------------------
    # Model
    # -------------------------------
    print("Loading model...")

    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 38)

    model = model.to(device)

    # Freeze backbone (important for CPU)
    for param in model.features.parameters():
        param.requires_grad = False

    print("Model ready")

    # -------------------------------
    # Loss & Optimizer
    # -------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -------------------------------
    # Training
    # -------------------------------
    epochs = 5

    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch+1}...")

        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

        # Save model
        torch.save(model.state_dict(), f"models/model_epoch_{epoch+1}.pth")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()