from math import exp
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from copy import deepcopy

x_data = [0, 11, 17, 26]
y_data = [0.226, 0.236, 0.266, 0.306]

b = 0
w = 1

def neuron (x, b, w):
    affine = w*x + b
    return 1 / (1+exp(-affine))

def dndb (x):
    return exp(-w*x-b) / (1 + exp(-w*x-b))**2

def dndw (x):
    return ( exp(-w*x-b) / (1 + exp(-w*x-b))**2 ) *x

def loss (b,w):
    return (1/len(x_data)) * sum(map(lambda x,y: (x-y)**2, [neuron(x, b, w) for x in x_data], y_data))
loss_ = np.vectorize(loss)

def gradloss ():
    dw = (1/len(x_data)) * sum(map( lambda x,y: 2*(neuron(x,b,w)-y)*dndw(x), x_data, y_data))
    db = (1/len(x_data)) * sum(map( lambda x,y: 2*(neuron(x,b,w)-y)*dndb(x), x_data, y_data))
    return dw, db

l = 0.05
stepnum = 10000
beta = 0.9
epsilon = 10**(-8)

biases = []
weights = []
losses = []

#gradient descent with RMSProp
plt.title(str(stepnum)+" steps with learning rate "+str(l))
plt.xlabel("Years since 1995")
plt.ylabel("Fraction of wealth owned by top 1%")
m_w = 0
m_b = 0
for i in range(stepnum):
    plt.plot(range(27), [neuron(x,b,w) for x in range(27)], color="gray", ls="dashed")
    grad = gradloss()
    m_w = beta*m_w + (1-beta)*grad[0]*grad[0]
    m_b = beta*m_b + (1-beta)*grad[1]*grad[1]
    delta_w = -(l/(epsilon+(m_w)**0.5))*grad[0]
    delta_b = -(l/(epsilon+(m_b)**0.5))*grad[1]
    w += delta_w
    b += delta_b
    biases.append(deepcopy(b))
    weights.append(deepcopy(w))
    losses.append(loss(b,w))
#plt.ylim((0,1))
plt.plot(np.linspace(0,27,100), [neuron(x,b,w) for x in np.linspace(0,27,100)], color="black")
plt.scatter(x_data, y_data, color="black", zorder=100)
plt.show()

plt.title(str(stepnum)+" steps with learning rate "+str(l))
plt.ylabel("Mean squared error loss")
plt.xlabel("Step number")
plt.plot(losses, color="black")
plt.show()

#plot loss
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X,Y = np.meshgrid(x, y)
Z = loss_(X,Y)
#fig = plt.figure(figsize = (10,7))
ax = plt.axes(projection='3d')
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='none')
ax.set_title("Loss landscape")
ax.set_xlabel('Bias')
ax.set_ylabel('Weight')
ax.set_zlabel('Loss')
ax.plot(biases, weights, losses, zorder=100, color="black")

