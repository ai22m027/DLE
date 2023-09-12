#plotting
import matplotlib.pyplot as plt
#deep learning
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
#helper
from collections import OrderedDict
#rng
from random import random


#data generation
def myfunction (tup):
    """
    Binary classification in quadrants.
    """
    if tup[0] < 0 and tup[1] > 0:
        return 0
    if tup[0] > 0 and tup[1] > 0:
        return 1
    if tup[0] < 0 and tup[1] < 0:
        return 1
    return 0

#data
x1_train = [2*random()-1 for i in range(1000)]
x2_train = [2*random()-1 for i in range(1000)]
x_train = zip(x1_train, x2_train)
x_train = [list(x) for x in x_train]
#coerce to torch tensors
X_train = torch.Tensor(x_train)
y_train = [myfunction(tup) for tup in x_train]
#take care to make single numbers arrays
Y_train = torch.Tensor([[yy] for yy in y_train])

def plot_nicely (x1, x2, y, title=""):
    """
    Visualise the training data in 2D.
    Cartesian coordinates correspond to feature variables.
    Coloring corresponds to the target variable.
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
    plt.show()

#Look at the data
plot_nicely(x1_train, x2_train, y_train, title="Training data")


def make_torch_net (input_length:int=2, width:int=2, output_length:int=1, hidden=3):
    """
    Make a torch network that takes a vector (actually a torch tensor) of length 'input_length',
    has 'hidden' hidden layers with ReLU-activation, 
    and returns a vector (actually a torch tensor) of length 'output_length'.
    """
    layers = []
    layer_num = 0
    layers.append((str(layer_num), nn.Linear(input_length, width)))
    layer_num += 1
    layers.append((str(layer_num), nn.Tanh()))
    layer_num += 1
    for i in range(hidden):
        layers.append((str(layer_num), nn.Linear(width,width)))
        layer_num += 1
        layers.append((str(layer_num), nn.Tanh()))
        layer_num += 1
    layers.append((str(layer_num), nn.Linear(width, output_length)))
    layer_num += 1
    layers.append((str(layer_num), nn.Sigmoid()))
    net = nn.Sequential(OrderedDict(layers))
    print(net)
    return net


class Net (nn.Module):
    """
    Torch network with one hidden layer of four tanh-activated neurons
    and an output layer of a single logistically-activated neuron.
    """
    def __init__ (self):
        super().__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(2,4), nn.Tanh())
        self.output_layer = nn.Sequential(nn.Linear(4,1), nn.Sigmoid())
    def forward (self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

#initialite torch net
net = Net()

### training
loss_fn = binary_cross_entropy
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)

losses=[]
for e in range(3000):
    if e%100 ==0:
        print("Episode number "+str(e))
    # Compute prediction and loss
    optimizer.zero_grad()
    pred = net(X_train)
    loss = loss_fn(pred, Y_train)
    losses.append(loss.item())
    # Backpropagation
    loss.backward()
    optimizer.step()

plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylim((0,1))
plt.plot(losses, color="black")
plt.show()
###

#get predictions of network on the training data
pred = net(X_train)
#plot the predictions
plot_nicely(x1_train, x2_train, pred.detach().numpy(), title="Predictions on training data")

#get the output of the hidden layer
pred_hidden = net.hidden_layer(X_train)
#normalize output of hidden layer for visualisation
pred_hidden = pred_hidden - pred_hidden.min()
pred_hidden = pred_hidden / pred_hidden.max()

#visualise outputs of hidden layer
plot_nicely(x1_train, x2_train, [x[0].item() for x in pred_hidden], title="First hidden neuron")
plot_nicely(x1_train, x2_train, [x[1].item() for x in pred_hidden], title="Second hidden neuron")
plot_nicely(x1_train, x2_train, [x[2].item() for x in pred_hidden], title="Third hidden neuron")
plot_nicely(x1_train, x2_train, [x[3].item() for x in pred_hidden], title="Fourth hidden neuron")

