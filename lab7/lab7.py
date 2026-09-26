# Import PyTorch
import torch
# Import neural network modules
from torch import nn
# Import DataLoader
from torch.utils.data import DataLoader
# Import MNIST dataset and transforms
from torchvision import datasets,transforms

# Select GPU if available otherwise CPU
device="cuda" if torch.cuda.is_available() else "cpu"
print("Using:",device)

# Convert MNIST images into tensors
transform=transforms.ToTensor()

# Load training data from local folder
train_data=datasets.MNIST(root="./data",train=True,download=False,transform=transform)

# Load testing data from local folder
test_data=datasets.MNIST(root="./data",train=False,download=False,transform=transform)

# Create batches for training
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)

# Create batches for testing
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)

# Define deep feedforward neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        # Initialize parent class
        super().__init__()
        # Define network layers
        self.flatten=nn.Flatten()
        self.network=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        # Flatten input image
        x=self.flatten(x)
        # Pass input through network
        return self.network(x)

# Create model and move it to CPU/GPU
model=NeuralNetwork().to(device)
print(model)

# Define loss function
loss_fn=nn.CrossEntropyLoss()

# Define optimizer and learning rate
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

# Define training function
def train(dataloader,model,loss_fn,optimizer):
    # Set model to training mode
    model.train()
    total_loss=0

    # Process data batch by batch
    for X,y in dataloader:
        # Move data to selected device
        X,y=X.to(device),y.to(device)

        # Forward pass
        pred=model(X)

        # Calculate loss
        loss=loss_fn(pred,y)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss+=loss.item()

    # Return average loss
    return total_loss/len(dataloader)

# Define testing function
def test(dataloader,model,loss_fn):
    # Set model to evaluation mode
    model.eval()
    test_loss=0
    correct=0
    size=len(dataloader.dataset)

    # Disable gradient calculation during testing
    with torch.no_grad():
        for X,y in dataloader:
            # Move data to selected device
            X,y=X.to(device),y.to(device)

            # Make predictions
            pred=model(X)

            # Calculate loss
            test_loss+=loss_fn(pred,y).item()

            # Count correct predictions
            correct+=(pred.argmax(1)==y).sum().item()

    # Calculate average loss
    test_loss/=len(dataloader)

    # Calculate accuracy
    accuracy=correct/size

    print("Test Accuracy:",100*accuracy,"%")
    print("Test Loss:",test_loss)

    return test_loss,accuracy

# Number of complete passes through training data
epochs=10

# Train model
for epoch in range(epochs):
    print("Epoch:",epoch+1)

    # Train model using training batches
    train_loss=train(train_loader,model,loss_fn,optimizer)

    # Test model after each epoch
    test_loss,accuracy=test(test_loader,model,loss_fn)

    print("Training Loss:",train_loss)

# Save trained model
torch.save(model.state_dict(),"lab7_model.pth")
print("Model saved")

# Create new model
loaded_model=NeuralNetwork().to(device)

# Load saved parameters
loaded_model.load_state_dict(torch.load("lab7_model.pth",weights_only=True))

# Set loaded model to evaluation mode
loaded_model.eval()

print("Model loaded successfully")