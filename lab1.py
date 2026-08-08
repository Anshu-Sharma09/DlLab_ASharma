import numpy as np
import matplotlib.pyplot as plt
z=np.linspace(-10,10,100)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoid_derivative(z):
    s=sigmoid(z)
    return s*(1-s)
def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
def tanh_derivative(z):
    t=tanh(z)
    return 1-t**2
def relu(z):
    y=np.zeros(len(z))
    for i in range(len(z)):
        if z[i]>0:
            y[i]=z[i]
        else:
            y[i]=0
    return y
def relu_derivative(z):
    y=np.zeros(len(z))
    for i in range(len(z)):
        if z[i]>0:
            y[i]=1
        else:
            y[i]=0
    return y
def leaky_relu(z):
    y=np.zeros(len(z))
    for i in range(len(z)):
        if z[i]>0:
            y[i]=z[i]
        else:
            y[i]=0.01*z[i]
    return y
def leaky_relu_derivative(z):
    y=np.zeros(len(z))
    for i in range(len(z)):
        if z[i]>0:
            y[i]=1
        else:
            y[i]=0.01
    return y
def softmax(z):
    exp_z=np.exp(z-np.max(z))
    return exp_z/np.sum(exp_z)
def softmax_derivative(z):
    s=softmax(z)
    n=len(s)
    derivative=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                derivative[i,j]=s[i]*(1-s[i])
            else:
                derivative[i,j]=-s[i]*s[j]
    return derivative
sigmoid_output=sigmoid(z)
sigmoid_grad=sigmoid_derivative(z)
tanh_output=tanh(z)
tanh_grad=tanh_derivative(z)
relu_output=relu(z)
relu_grad=relu_derivative(z)
leaky_output=leaky_relu(z)
leaky_grad=leaky_relu_derivative(z)
print("Softmax output:")
print(softmax(z))
print("\nSoftmax derivative:")
print(softmax_derivative(z))
plt.figure()
plt.plot(z,sigmoid_output)
plt.title("Sigmoid")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid()
plt.show()
plt.figure()
plt.plot(z,sigmoid_grad)
plt.title("Sigmoid Derivative")
plt.xlabel("z")
plt.ylabel("Gradient")
plt.grid()
plt.show()
plt.figure()
plt.plot(z,tanh_output)
plt.title("Tanh")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid()
plt.show()
plt.figure()
plt.plot(z,tanh_grad)
plt.title("Tanh Derivative")
plt.xlabel("z")
plt.ylabel("Gradient")
plt.grid()
plt.show()
plt.figure()
plt.plot(z,relu_output)
plt.title("ReLU")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid()
plt.show()
plt.figure()
plt.plot(z,relu_grad)
plt.title("ReLU Derivative")
plt.xlabel("z")
plt.ylabel("Gradient")
plt.grid()
plt.show()
plt.figure()
plt.plot(z,leaky_output)
plt.title("Leaky ReLU")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid()
plt.show()
plt.figure()
plt.plot(z,leaky_grad)
plt.title("Leaky ReLU Derivative")
plt.xlabel("z")
plt.ylabel("Gradient")
plt.grid()
plt.show()