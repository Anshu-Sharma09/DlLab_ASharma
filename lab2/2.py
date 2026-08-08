import numpy as np
#for 1st network
np.random.seed(42)
def sigmoid(z):
    return 1/(1+np.exp(-z))
x=np.array([[0.5],[0.2],[0.1],[0.9]])
W=np.random.randn(1,4)*0.1
b=np.random.randn(1,1)*0.1
y=np.array([[1]])
z=W@x+b
a=sigmoid(z)
y_pred=a
l=0.5*np.sum((y-y_pred)**2)
print("z (output layer):",z)
print("a (output layer):",a)
print("y_pred:",y_pred)
print("Loss:",l)


#for 2nd network:

x=np.array([[0.5],[0.2],[0.1],[0.9]])
W1=np.random.randn(3,4)*0.1
b1=np.random.randn(3,1)*0.1
W2=np.random.randn(2,3)*0.1
b2=np.random.randn(2,1)*0.1
W3=np.random.randn(1,2)*0.1
b3=np.random.randn(1,1)*0.1
y=np.array([[1]])
z1=W1@x+b1
a1=sigmoid(z1)
z2=W2@a1+b2
a2=sigmoid(z2)
z3=W3@a2+b3
a3=sigmoid(z3)
y_pred=a3
l=0.5*np.sum((y-y_pred)**2)
print("z1 (hidden layer 1):",z1)
print("a1 (hidden layer 1):",a1)
print("z2 (hidden layer 2):",z2)
print("a2 (hidden layer 2):",a2)
print("z3 (output layer):",z3)
print("a3 (output layer):",a3)
print("y_pred:",y_pred)
print("Loss:",l)