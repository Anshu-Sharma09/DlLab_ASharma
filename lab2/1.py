import numpy as np
#for 1st network
np.random.seed(42)
def sigmoid(z):
    return 1/(1+np.exp(-z))
x=np.array([0.5,0.2,0.1,0.9])
W=np.random.randn(4)*0.1
b=np.random.randn()*0.1
y=1
z=W[0]*x[0]+W[1]*x[1]+W[2]*x[2]+W[3]*x[3]+b
a=sigmoid(z)
y_hat=a
l=0.5*(y-y_hat)**2
print("z =",z)
print("a =",a)
print("y_hat =",y_hat)
print("Loss =",l)


#for 2nd networkk import numpy as np

x=np.array([0.5,0.2,0.1,0.9])
W1=np.random.randn(3,4)*0.1
b1=np.random.randn(3)*0.1
W2=np.random.randn(2,3)*0.1
b2=np.random.randn(2)*0.1
W3=np.random.randn(1,2)*0.1
b3=np.random.randn()*0.1
y=1
z1_1=W1[0,0]*x[0]+W1[0,1]*x[1]+W1[0,2]*x[2]+W1[0,3]*x[3]+b1[0]
z1_2=W1[1,0]*x[0]+W1[1,1]*x[1]+W1[1,2]*x[2]+W1[1,3]*x[3]+b1[1]
z1_3=W1[2,0]*x[0]+W1[2,1]*x[1]+W1[2,2]*x[2]+W1[2,3]*x[3]+b1[2]
a1_1=sigmoid(z1_1)
a1_2=sigmoid(z1_2)
a1_3=sigmoid(z1_3)
z2_1=W2[0,0]*a1_1+W2[0,1]*a1_2+W2[0,2]*a1_3+b2[0]
z2_2=W2[1,0]*a1_1+W2[1,1]*a1_2+W2[1,2]*a1_3+b2[1]
a2_1=sigmoid(z2_1)
a2_2=sigmoid(z2_2)
z3=W3[0,0]*a2_1+W3[0,1]*a2_2+b3
a3=sigmoid(z3)
y_pred=a3
l=0.5*(y-y_hat)**2
print("Layer 1 activation:")
print(a1_1,a1_2,a1_3)
print("Layer 2 activation:")
print(a2_1,a2_2)
print("Output activation:")
print(a3)
print("y_hat =",y_pred)
print("Loss =",l)