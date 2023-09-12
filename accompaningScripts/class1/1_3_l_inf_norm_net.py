import numpy as np
import matplotlib.pyplot as plt
#deep learning
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
#rng
from random import random


#data generation
def myfunction (tup):
    """
    L-infinity norm on R^2.
    """
    return max(abs(tup[0]), abs(tup[1]))
def myfunction_plotting (x1, x2):
    """
    Convenience function for plotting.
    """
    return max(abs(x1), abs(x2))

#generate training data
x1_train = [2*random()-1 for i in range(1000)]
x2_train = [2*random()-1 for i in range(1000)]
x_train = zip(x1_train, x2_train)
x_train = [list(x) for x in x_train]
#coerce t torch tensor class
X_train = torch.Tensor(x_train)
y_train = [myfunction(tup) for tup in x_train]
#take care to make single numbers arrays
Y_train = torch.Tensor([[yy] for yy in y_train])

def plot_2D (x1, x2, y, title=""):
    """
    Plot the training data in 2D.
    The cartesian coordinates correspond to the features.
    The coloring corresponds to the target values.
    """
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect(1)
    ax.set_xlim((min(x1),max(x1)))
    ax.set_ylim((min(x2),max(x2)))
    ax.scatter(x1, x2, c=y, s=20, cmap="gray", edgecolor="black")
    fig.show()

def plot_func (func, title=""):
    """
    Plot the graph of the data-generating function in 3D.
    """
    x1 = np.linspace(-1,1,100)
    x2 = np.linspace(-1,1,100)
    X1,X2 = np.meshgrid(x1, x2)
    func_ = np.vectorize(func)
    Z = func_(X1,X2)
    ax = plt.axes(projection='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.plot_wireframe(X1, X2, Z, rstride=3, cstride=3, color="black")
    #ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.gist_gray, edgecolor='none')
    #ax.plot_surface(X1, X2, Z, cmap=cm.gist_gray, edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')
    plt.show()

#the graph of the data-generating function
plot_func(myfunction_plotting, title="Data-generating function")

#2D visualisation of the training data
plot_2D(x1_train, x2_train, y_train, title="Training data")


class Net (nn.Module):
    """
    Class to create torch networks for the problem.
    One tanh-activated hidden layer.
    Logistically-acticated output layer.
    The width of the hidden layer can be adjusted.
    """
    def __init__ (self, width:int):
        super().__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(2,width), nn.Tanh())
        self.output_layer = nn.Sequential(nn.Linear(width,1), nn.Sigmoid())
    def forward (self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

class MinNet (nn.Module):
    """
    This neural network is theoretically sufficient to exactly compute the function (see slides).
    Can we find the correct parameters by training?
    """
    def __init__ (self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(2,4), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(4,2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(2,2), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(2,1), nn.ReLU())
    def forward (self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


#net = Net(width=4)
#We know that in theory the following net should suffice
net = MinNet()

### training
loss_fn = mse_loss
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)

losses=[]
for e in range(10000):
    if e%100 ==0:
        print("Epoch number "+str(e))
    # Compute prediction and loss
    optimizer.zero_grad()
    pred = net(X_train)
    loss = loss_fn(pred, Y_train)
    losses.append(loss.item())
    if loss < 0.005:
        print("Epochs needed: "+str(e))
        break
    # Backpropagation
    loss.backward()
    optimizer.step()

def plot_losses (losses):
    fig, ax = plt.subplots()
    ax.set_title("Loss evolution")
    ax.set_ylim((0,max(losses)))
    ax.plot(losses, color="black")
    fig.show()
    plt.show()

plot_losses(losses)
###


def plot_net (net, title=""):
    """
    Plot the graph of the neural network.
    """
    x1 = np.linspace(-1,1,100)
    x2 = np.linspace(-1,1,100)
    X1,X2 = np.meshgrid(x1, x2)
    
    Z = np.zeros(shape=(100,100))
    for i in range(100):
        for j in range(100):
            Z[i][j] = net(torch.Tensor([x1[i],x2[j]])).item()
    
    ax = plt.axes(projection='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.plot_wireframe(X1, X2, Z, rstride=3, cstride=3, color="black")
    #ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.gist_gray, edgecolor='none')
    #ax.plot_surface(X1, X2, Z, cmap=cm.gist_gray, edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Trained neural net')
    plt.show()

plot_net(net=net)

#print parameter values
#list(net.layer2.parameters())
