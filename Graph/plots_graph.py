import matplotlib.pyplot as plt

# Your recorded values
epochs = [1, 2, 3, 4, 5]
accuracy = [83.33, 88.97, 91.17, 92.11, 93.09]

# Approx loss (optional – based on your logs)
loss = [2339, 1044, 710, 565, 474]

# Accuracy graph
plt.figure()
plt.plot(epochs, accuracy)
plt.title("Validation Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.savefig("accuracy.png")

# Loss graph
plt.figure()
plt.plot(epochs, loss)
plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.png")

print("Graphs created!")