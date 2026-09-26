# Import libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

# Select GPU if available otherwise CPU
device="cuda" if torch.cuda.is_available() else "cpu"
print("Using:",device)

# Convert images to tensors
transform=transforms.ToTensor()

# Load MNIST data
train_data=datasets.MNIST(root="../lab7/data",train=True,download=False,transform=transform)
test_data=datasets.MNIST(root="../lab7/data",train=False,download=False,transform=transform)

# Create DataLoaders
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)

# Batch normalization from scratch
def batch_normalization(x,eps=1e-5):
    # Calculate mean across batch
    mean=x.mean(dim=0,keepdim=True)
    # Calculate variance across batch
    variance=x.var(dim=0,unbiased=False,keepdim=True)
    # Normalize input
    return (x-mean)/torch.sqrt(variance+eps)

x=torch.randn(4,3)
print("Input:",x)
print("Batch Normalized:",batch_normalization(x))

# Layer normalization from scratch
def layer_normalization(x,eps=1e-5):
    # Calculate mean for each sample
    mean=x.mean(dim=1,keepdim=True)
    # Calculate variance for each sample
    variance=x.var(dim=1,unbiased=False,keepdim=True)
    # Normalize each sample
    return (x-mean)/torch.sqrt(variance+eps)

print("Layer Normalized:",layer_normalization(x))

# Dropout from scratch
def dropout(x,p=0.5,training=True):
    # Return unchanged input during testing
    if not training:
        return x
    # Create random dropout mask
    mask=(torch.rand_like(x)>p).float()
    # Scale remaining values
    return x*mask/(1-p)

x=torch.ones(10)
print("Dropout:",dropout(x))

# Network with batch normalization
class BatchNormNetwork(nn.Module):
    def __init__(self):
        # Initialize parent class
        super().__init__()
        # Define network
        self.flatten=nn.Flatten()
        self.network=nn.Sequential(
            nn.Linear(784,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        # Flatten image and pass through network
        return self.network(self.flatten(x))

# Network with dropout
class DropoutNetwork(nn.Module):
    def __init__(self):
        # Initialize parent class
        super().__init__()
        # Define network
        self.flatten=nn.Flatten()
        self.network=nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,10)
        )
    def forward(self,x):
        # Flatten image and pass through network
        return self.network(self.flatten(x))

# Define loss function
loss_fn=nn.CrossEntropyLoss()

# Training function
def train(model,optimizer):
    # Set training mode
    model.train()
    total_loss=0
    # Train using batches
    for X,y in train_loader:
        # Move data to CPU/GPU
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
    return total_loss/len(train_loader)

# Testing function
def test(model):
    # Set evaluation mode
    model.eval()
    correct=0
    total=0
    # Disable gradients
    with torch.no_grad():
        for X,y in test_loader:
            # Move data to CPU/GPU
            X,y=X.to(device),y.to(device)
            # Make predictions
            pred=model(X)
            # Count correct predictions
            correct+=(pred.argmax(1)==y).sum().item()
            total+=y.size(0)
    accuracy=100*correct/total
    print("Accuracy:",accuracy)
    return accuracy

# SGD
sgd_model=BatchNormNetwork().to(device)
sgd_optimizer=torch.optim.SGD(sgd_model.parameters(),lr=0.01)

for epoch in range(3):
    loss=train(sgd_model,sgd_optimizer)
    print("SGD Epoch:",epoch+1,"Loss:",loss)
test(sgd_model)

# SGD with Momentum
momentum_model=BatchNormNetwork().to(device)
momentum_optimizer=torch.optim.SGD(momentum_model.parameters(),lr=0.01,momentum=0.9)

for epoch in range(3):
    loss=train(momentum_model,momentum_optimizer)
    print("Momentum Epoch:",epoch+1,"Loss:",loss)
test(momentum_model)

# AdaGrad
adagrad_model=BatchNormNetwork().to(device)
adagrad_optimizer=torch.optim.Adagrad(adagrad_model.parameters(),lr=0.01)

for epoch in range(3):
    loss=train(adagrad_model,adagrad_optimizer)
    print("AdaGrad Epoch:",epoch+1,"Loss:",loss)
test(adagrad_model)

# Adam
adam_model=BatchNormNetwork().to(device)
adam_optimizer=torch.optim.Adam(adam_model.parameters(),lr=0.001)

for epoch in range(3):
    loss=train(adam_model,adam_optimizer)
    print("Adam Epoch:",epoch+1,"Loss:",loss)
test(adam_model)

# Dropout network with Adam
dropout_model=DropoutNetwork().to(device)
dropout_optimizer=torch.optim.Adam(dropout_model.parameters(),lr=0.001)

for epoch in range(3):
    loss=train(dropout_model,dropout_optimizer)
    print("Dropout Epoch:",epoch+1,"Loss:",loss)
test(dropout_model)