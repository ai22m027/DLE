from math import exp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
#from matplotlib import cm

#data from world inequality database
x_data = [0, 11, 17, 26] #years since 1997
y_data = [0.226, 0.236, 0.266, 0.306] #percentage of Austrian wealth owned by top 1%

#starting parameters of neuron
b = 0
w = 1

def neuron (x:float, b:float, w:float):
    """
    One-dimensional neuron with logistic activation.
    """
    affine = w*x + b
    return 1 / (1+exp(-affine))

def dndb (x):
    """
    Derivative with respect to the bias.
    """
    return exp(-w*x-b) / (1 + exp(-w*x-b))**2

def dndw (x):
    """
    Derivative with respect to the weight.
    """
    return ( exp(-w*x-b) / (1 + exp(-w*x-b))**2 ) *x

def loss (b,w):
    """
    Mean squared error loss.
    """
    return (1/len(x_data)) * sum(map(lambda x,y: (x-y)**2, [neuron(x, b, w) for x in x_data], y_data))
#numpy-vectorized version of mean squared error
loss_ = np.vectorize(loss)

def gradloss ():
    """
    Gradient of mean squared error.
    """
    dw = (1/len(x_data)) * sum(map( lambda x,y: 2*(neuron(x,b,w)-y)*dndw(x), x_data, y_data))
    db = (1/len(x_data)) * sum(map( lambda x,y: 2*(neuron(x,b,w)-y)*dndb(x), x_data, y_data))
    return dw, db

#parameters of gradient descent
l = 0.05 #learning rate
stepnum = 10000 #number of steps

#to store the history
biases = []
weights = []
losses = []

#vanilla gradient descent
plt.title(str(stepnum)+" steps with learning rate "+str(l))
plt.xlabel("Years since 1995")
plt.ylabel("Fraction of wealth owned by top 1%")
for i in range(stepnum):
    plt.plot(range(27), [neuron(x,b,w) for x in range(27)], color="gray", ls="dashed")
    grad = gradloss()
    delta_w = -l*grad[0]
    delta_b = -l*grad[1]
    w += delta_w
    b += delta_b
    biases.append(deepcopy(b))
    weights.append(deepcopy(w))
    losses.append(loss(b,w))
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
ax = plt.axes(projection='3d')
#ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, color="gray")
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.gray, edgecolor='none')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='none')
ax.set_title("Loss landscape")
ax.set_xlabel('Bias')
ax.set_ylabel('Weight')
ax.set_zlabel('Loss')
ax.plot(biases, weights, losses, zorder=100, color="black")
plt.show()


#plot final model with data
plt.xticks(x_data, [1997, 2008, 2014, 2023])
plt.xlabel("Year")
plt.title("Fraction of Austrian wealth owned by top 1%")
plt.plot(np.linspace(0,27,100), [neuron(x,b,w) for x in np.linspace(0,27,100)], color="black")
plt.scatter(x_data, y_data, marker="x", color="black", zorder=100)
plt.show()
