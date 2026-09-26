import numpy as np
import matplotlib.pyplot as plt
X=np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]],dtype=float)
y=np.array([[0],[1],[1],[0]],dtype=float)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoid_derivative(a):
    return a*(1-a)
np.random.seed(42)
W=np.random.randn(3,1)
b=np.zeros((1,1))
learning_rate=0.1
loss_history=[]
for iteration in range(1000):
    z=np.dot(X,W)+b
    y_pred=sigmoid(z)
    loss=np.mean((y-y_pred)**2)
    loss_history.append(loss)
    dL_dypred=2*(y_pred-y)/len(y)
    dy_pred_dz=sigmoid_derivative(y_pred)
    dL_dz=dL_dypred*dy_pred_dz
    dL_dW=np.dot(X.T,dL_dz)
    dL_db=np.sum(dL_dz,axis=0,keepdims=True)
    W=W-learning_rate*dL_dW
    b=b-learning_rate*dL_db
    if iteration%100==0:
        print("Iteration:",iteration,"Loss:",loss)
print("\nFinal Predictions:")
print(y_pred)
print("\nRounded Predictions:")
print((y_pred>=0.5).astype(int))
print("\nFinal Weights:")
print(W)
print("\nFinal Bias:")
print(b)
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()