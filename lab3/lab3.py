import numpy as np
#for 1st network
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(a):
    return a*(1-a)
X=np.array([0.5,0.3,0.2,0.7])
y=1
W=np.array([0.4,0.2,-0.3,0.1])
b=0.5
z=np.dot(X,W)+b
a=sigmoid(z)
loss=0.5*(y-a)**2
print("Prediction =",a)
print("Loss =",loss)
dL_da=a-y
da_dz=sigmoid_derivative(a)
delta=dL_da*da_dz
dW=delta*X
db=delta
print("Gradient at Output Neuron")
print(delta)
print("Weight Gradients")
for i in range(4):
    print(f"dW{i+1} =",dW[i])
print("db =",db)

#for 2nd network

X=np.array([[0.5],[0.3],[0.2],[0.7]])
y=1
np.random.seed(0)
W1=np.random.randn(3,4)
b1=np.random.randn(3,1)
W2=np.random.randn(2,3)
b2=np.random.randn(2,1)
W3=np.random.randn(1,2)
b3=np.random.randn(1,1)
z1=W1@X+b1
a1=sigmoid(z1)
z2=W2@a1+b2
a2=sigmoid(z2)
z3=W3@a2+b3
a3=sigmoid(z3)
loss=0.5*(y-a3)**2
print("Prediction =",a3[0][0])
print("Loss =",loss[0][0])
delta3=(a3-y)*sigmoid_derivative(a3)
dW3=delta3@a2.T
db3=delta3
delta2=(W3.T@delta3)*sigmoid_derivative(a2)
dW2=delta2@a1.T
db2=delta2
delta1=(W2.T@delta2)*sigmoid_derivative(a1)
dW1=delta1@X.T
db1=delta1
print("Layer 3")
print("Neuron Gradient",delta3)
print("Weight Gradients",dW3)
print("Bias Gradient",db3)
print("Layer 2")
for i in range(2):
    print(f"Neuron {i+1} Gradient =",delta2[i][0])
print("Weight Gradients",dW2)
print("Bias Gradients",db2)
print("Layer 1")
for i in range(3):
    print(f"Neuron {i+1} Gradient =",delta1[i][0])
print("Weight Gradients",dW1)
print("Bias Gradient",db1)
