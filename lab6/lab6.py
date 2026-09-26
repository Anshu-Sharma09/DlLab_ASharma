# Import PyTorch
import torch
# Import neural network modules
from torch import nn
# Import optimization
from torch import optim
# Import datasets and transforms
from torchvision import datasets,transforms
# Import DataLoader and Dataset
from torch.utils.data import DataLoader,Dataset

# Check PyTorch installation
print("PyTorch:",torch.__version__)
import torchvision
print("Torchvision:",torchvision.__version__)
print("CUDA:",torch.cuda.is_available())

# Create a tensor
x=torch.tensor([[1,2],[3,4]])
print(x)
print(x.shape)
print(x.dtype)
print(x.ndim)

# Create zero tensor
zeros=torch.zeros(2,3)
print(zeros)

# Create one tensor
ones=torch.ones(2,3)
print(ones)

# Create random tensor
random_tensor=torch.rand(2,3)
print(random_tensor)

# Create tensor using range
range_tensor=torch.arange(10)
print(range_tensor)

# Tensor operations
a=torch.tensor([1,2,3])
b=torch.tensor([4,5,6])
print(a+b)
print(a*b)
print(a.sum())

# Matrix multiplication
m=torch.tensor([[1,2],[3,4]])
n=torch.tensor([[1,0],[0,1]])
print(torch.matmul(m,n))

# TensorDataset and DataLoader
from torch.utils.data import TensorDataset
simple_dataset=TensorDataset(a,b)
simple_loader=DataLoader(simple_dataset,batch_size=2,shuffle=True)
for batch_x,batch_y in simple_loader:
    print(batch_x)
    print(batch_y)
    break

# Convert images to tensors
transform=transforms.ToTensor()

# Load local MNIST training dataset
train_dataset=datasets.MNIST(root="./data",train=True,download=False,transform=transform)

# Load local MNIST test dataset
test_dataset=datasets.MNIST(root="./data",train=False,download=False,transform=transform)

# Check dataset and image
print(len(train_dataset))
image,label=train_dataset[0]
print(image.shape)
print(label)

# Create DataLoaders
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

# Define neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        # Initialize parent class
        super().__init__()
        # Flatten 28x28 image to 784 values
        self.flatten=nn.Flatten()
        # Define network layers
        self.network=nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        # Flatten input
        x=self.flatten(x)
        # Pass through network
        return self.network(x)

# Create model
model=NeuralNetwork()
print(model)

# Test forward pass
sample_image=image.unsqueeze(0)
prediction=model(sample_image)
print(prediction)
print(prediction.shape)

# Define loss function
loss_fn=nn.CrossEntropyLoss()
loss=loss_fn(prediction,torch.tensor([label]))
print(loss)

# Demonstrate autograd
x=torch.tensor(2.0,requires_grad=True)
y=x**2
y.backward()
print(x.grad)

# Create optimizer
optimizer=optim.Adam(model.parameters(),lr=0.001)

# Train model
for epoch in range(5):
    # Training mode
    model.train()
    running_loss=0.0
    for images,labels in train_loader:
        # Clear old gradients
        optimizer.zero_grad()
        # Forward pass
        outputs=model(images)
        # Calculate loss
        loss=loss_fn(outputs,labels)
        # Calculate gradients
        loss.backward()
        # Update weights
        optimizer.step()
        running_loss+=loss.item()
    average_loss=running_loss/len(train_loader)
    print("Epoch:",epoch+1,"Loss:",average_loss)

# Evaluate model
model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        # Predict classes
        outputs=model(images)
        predicted=torch.argmax(outputs,dim=1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

# Calculate accuracy
accuracy=100*correct/total
print("Accuracy:",accuracy)

# Save model
torch.save(model.state_dict(),"model.pth")
print("Model saved")

# Load model
loaded_model=NeuralNetwork()
loaded_model.load_state_dict(torch.load("model.pth",weights_only=True))
loaded_model.eval()
print("Model loaded successfully")
print(loaded_model)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self,data,labels,transform=None):
        # Store data
        self.data=data
        # Store labels
        self.labels=labels
        # Store transform
        self.transform=transform
    def __len__(self):
        # Return dataset size
        return len(self.data)
    def __getitem__(self,index):
        # Get image and label
        image=self.data[index]
        label=self.labels[index]
        # Apply transform if provided
        if self.transform:
            image=self.transform(image)
        return image,label

# Create small custom dataset from MNIST
custom_data=[train_dataset[i][0] for i in range(100)]
custom_labels=[train_dataset[i][1] for i in range(100)]

# Create custom Dataset
custom_dataset=CustomDataset(custom_data,custom_labels)

# Create DataLoader
custom_loader=DataLoader(custom_dataset,batch_size=10,shuffle=True)

# Check custom dataset
print("Custom dataset size:",len(custom_dataset))
custom_images,custom_labels=next(iter(custom_loader))
print("Custom batch shape:",custom_images.shape)
print("Custom batch labels:",custom_labels)