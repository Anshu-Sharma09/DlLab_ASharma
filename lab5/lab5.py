import numpy as np
import matplotlib.pyplot as plt
X=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
y=np.array([[0],[1],[1],[0]],dtype=float)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoid_derivative(a):
    return a*(1-a)
np.random.seed(42)
W1=np.random.randn(2,2)
b1=np.zeros((1,2))
W2=np.random.randn(2,1)
b2=np.zeros((1,1))
learning_rate=0.1
loss_history=[]
for iteration in range(10000):
    z1=np.dot(X,W1)+b1
    a1=sigmoid(z1)
    z2=np.dot(a1,W2)+b2
    y_pred=sigmoid(z2)
    loss=np.mean((y-y_pred)**2)
    loss_history.append(loss)
    dL_dypred=2*(y_pred-y)/len(y)
    dypred_dz2=sigmoid_derivative(y_pred)
    dL_dz2=dL_dypred*dypred_dz2
    dL_dW2=np.dot(a1.T,dL_dz2)
    dL_db2=np.sum(dL_dz2,axis=0,keepdims=True)
    dL_da1=np.dot(dL_dz2,W2.T)
    da1_dz1=sigmoid_derivative(a1)
    dL_dz1=dL_da1*da1_dz1
    dL_dW1=np.dot(X.T,dL_dz1)
    dL_db1=np.sum(dL_dz1,axis=0,keepdims=True)
    W2=W2-learning_rate*dL_dW2
    b2=b2-learning_rate*dL_db2
    W1=W1-learning_rate*dL_dW1
    b1=b1-learning_rate*dL_db1
    if iteration%1000==0:
        print("Iteration:",iteration,"Loss:",loss)
print("\nPredictions:")
print(y_pred)
print("\nRounded Predictions:")
print((y_pred>=0.5).astype(int))
print("\nW1:")
print(W1)
print("\nW2:")
print(W2)
print("\nBias 1:")
print(b1)
print("\nBias 2:")
print(b2)
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("XOR Training Loss")
plt.show()
