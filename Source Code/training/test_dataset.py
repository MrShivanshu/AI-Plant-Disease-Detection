from torchvision import datasets

dataset = datasets.ImageFolder("data/PlantVillage-Dataset/raw/color")

print("Total images:", len(dataset))
print("Sample classes:", dataset.classes[:5])