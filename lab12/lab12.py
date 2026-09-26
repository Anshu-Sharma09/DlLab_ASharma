import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------
# 1. DEVICE
# ---------------------------------------------------------
# CUDA is used if GPU is available; otherwise CPU is used.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ---------------------------------------------------------
# 2. LOAD MNIST DATASET
# ---------------------------------------------------------
# MNIST images are 28 x 28 grayscale images.
# 1 = one channel because the image is grayscale.
# ToTensor() converts pixel values from 0-255 to 0-1.
transform = transforms.ToTensor()

# Use the local dataset path if MNIST is already downloaded.
train_data = datasets.MNIST(
    root="../lab6/data",
    train=True,
    download=False,
    transform=transform
)

test_data = datasets.MNIST(
    root="../lab6/data",
    train=False,
    download=False,
    transform=transform
)

# 64 = batch size.
# It means 64 images are processed together before one update.
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ---------------------------------------------------------
# 3. CNN MODEL
# ---------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input: 1 x 28 x 28
            # 1 = input channel because MNIST is grayscale.
            # 32 = number of filters/features learned.
            # 3 = 3x3 convolution kernel.
            # padding=1 keeps height and width unchanged.
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            # 2x2 max pooling reduces spatial dimensions by half.
            # 28 x 28 -> 14 x 14
            nn.MaxPool2d(2),

            # Input channels = 32 because previous Conv2d produced 32 feature maps.
            # 64 = number of new filters/features.
            # 3 = 3x3 kernel.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # 14 x 14 -> 7 x 7
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            # After convolutions:
            # 64 feature maps x 7 x 7 = 3136 values.
            nn.Flatten(),

            # 3136 = 64 x 7 x 7.
            # 128 = chosen number of neurons in fully connected layer.
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            # MNIST has 10 classes: digits 0,1,2,...9.
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN().to(device)

# ---------------------------------------------------------
# 4. LOSS AND OPTIMIZER
# ---------------------------------------------------------
# CrossEntropyLoss is used for multi-class classification.
criterion = nn.CrossEntropyLoss()

# Adam is an optimization algorithm.
# 0.001 is a commonly used learning rate.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------
# 5. TRAINING
# ---------------------------------------------------------
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from previous iteration.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)

        # Calculate prediction error.
        loss = criterion(outputs, labels)

        # Calculate gradients.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Loss: {total_loss/len(train_loader):.4f}"
    )

# ---------------------------------------------------------
# 6. TESTING
# ---------------------------------------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Select class having the highest output.
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")