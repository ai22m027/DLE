from abc import ABC, abstractmethod
from math import exp
import numpy as np
from random import sample
import matplotlib.pyplot as plt

class Layer (ABC):
    """
    Abstract base class for layers of a neural network.
    """
    @abstractmethod
    def forward (self, x):
        #pass x forward and update 'last_input' and 'last_output'
        pass
    @abstractmethod
    def backward (self, y):
        #pass y backward and update gradient slots
        pass
    @abstractmethod
    def step (self, lr:float):
        #take a step in direction of gradients (with factor lr)
        pass
    @abstractmethod
    def zero_grad (self):
        #reset gradient slots (if they exist)
        pass

class NeuralNet:
    """
    A class for feedforward neural networks.
    """
    def __init__ (self):
        self.layers = []
    def add_layer (self, layer:Layer):
        """
        Adds the layer to the neural network (at the back). 
        Makes a compatibility check.
        """
        if self.layers and layer.dim_in != "any":
            for layer2 in reversed(self.layers):
                if layer2.dim_out != "any":
                    assert(layer2.dim_out == layer.dim_in), "The dimension of the new layer does not match."
                    break
        self.layers.append(layer)
    def forward (self, x):
        """
        Forward pass through the network.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward (self, y):
        """
        Backward pass through the network.
        """
        for layer in reversed(self.layers):
            y = layer.backward(y)
        return y
    def step (self, lr:float):
        """
        Take a step in the direction of the gradient slots.

        Parameters
        ----------
        lr : float
            Factor with which the entries in the gradient slots are multiplied.
            Should be positive.
        """
        for layer in self.layers:
            layer.step(lr)
    def zero_grad (self):
        """
        Reset all gradient slots to zero.
        """
        for layer in self.layers:
            layer.zero_grad()
    def print (self):
        """
        Print a short description of the network to standard output.
        """
        counter = 0
        string = "\n"
        for layer in self.layers:
            counter += 1
            string += str(counter) + ". " + layer.name + ": dim_in=" + str(layer.dim_in) + ", dim_out=" + str(layer.dim_out) + ".\n"
        print(string)

class Loss (ABC):
    """
    Abstract base class for loss functions on one example.
    """
    @abstractmethod
    def eval (self, true, predicted):
        #evaluate the loss on a single example
        pass
    @abstractmethod
    def derivative (self, true, predicted):
        #evaluate the derivative on a single example
        pass
    def on_sample (self, true_data, predicted_data):
        #compute mean loss over a sample
        assert(len(true_data) == len(predicted_data)), "The lengths of 'true_data' and 'predicted_data' differ."
        return sum([self.eval(true_data[i], predicted_data[i]) for i in range(len(true_data))]) / len(true_data)


class SELoss (Loss):
    """
    Squared error loss on an example:
        [e_1,e_2,...] -> e_1^2 + e_2^2 + ...
    """
    def eval (self, true, predicted):
        return sum(map(lambda a,b: (a-b)**2, true, predicted))
    def derivative (self, true, predicted):
        return list(map(lambda a,b: 2*(b-a), true, predicted))

class Logistic (Layer):
    """
    Logistic activation layer.
    Element-wise:
        x -> 1 / (1 + exp(-x))
    """
    def __init__ (self):
        self.name = "logistic"
        self.dim_in = "any"
        self.dim_out = "any"
        self.last_input = None
        self.last_output = None
    def zero_grad (self):
        #empty method since this layer has no parameters
        return None
    def step (self, lr:float):
        #empty method since this layer has no parameters
        return None
    def forward (self, x):
        x = np.array(x, dtype="float64")
        self.last_input = x
        self.last_output = 1 / (1 + np.exp(-x))
        #self.last_output = np.array([1 / (1 + exp(-xx)) for xx in x])
        return self.last_output
    def backward (self, y):
        y = np.array(y)
        return (self.last_output*(1-self.last_output))*y
        #return np.array([a*(1-a) for a in self.last_output], dtype="float64") * y
        #Above: clever way    
        #Below: straightforward way
        #return  np.array([exp(-x) / (1 + exp(-x)**2) for x in self.last_input], dtype="float64") * y

class ReLU (Layer):
    """
    Rectified linear units activation layer.
    Element-wise:
        x -> max(0,x)
    """
    def __init__ (self):
        self.name = "relu"
        self.dim_in = "any"
        self.dim_out = "any"
        self.last_input = None
        self.last_output = None
    def zero_grad (self):
        #empty method since this layer has no parameters
        return None
    def step (self, lr):
        #empty method since this layer has no parameters
        return None
    def forward (self, x):
        x = np.array(x, dtype="float64")
        self.last_input = x
        self.last_output = np.array([i if i > 0 else 0 for i in x], dtype="float64")
        return self.last_output
    def backward (self, y):
        y = np.array(y, dtype="float64")
        return np.array([1 if x >0 else 0 for x in self.last_input], dtype="float64") * y

class PReLU(Layer):
    """
    Parametric Leaky ReLU activation layer.
    Element-wise:
        x_i -> x_i if x_i > 0
        x_i -> a_i * x_i if x_i <= 0
    """

    def __init__(self, n):
        """
        Initialize the PReLU layer.

        Parameters
        ----------
        n : int
            Number of elements in the input vector.
        """
        self.name = "prelu"
        self.dim_in = n
        self.dim_out = n
        self.last_input = None
        self.last_output = None
        self.a = np.ones(n, dtype="float64")  # Initialize 'a' parameters to 1.0

    def zero_grad(self):
        """
        Reset gradient slots for 'a' to zero.
        """
        self.grad_a = np.zeros(self.dim_in, dtype="float64")

    def step(self, lr):
        """
        Take a step in the direction of the gradient for 'a' with learning rate 'lr'.

        Parameters
        ----------
        lr : float
            Learning rate for updating 'a' parameters.
        """
        self.a -= lr * self.grad_a

    def forward(self, x):
        """
        Forward pass through the PReLU layer.

        Parameters
        ----------
        x : np.ndarray
            Input vector.

        Returns
        -------
        np.ndarray
            Output vector after applying PReLU activation.
        """
        x = np.array(x, dtype="float64")
        self.last_input = x
        self.last_output = np.where(x > 0, x, self.a * x)
        return self.last_output

    def backward(self, y):
        """
        Backward pass through the PReLU layer.

        Parameters
        ----------
        y : np.ndarray
            Gradient from the next layer.

        Returns
        -------
        np.ndarray
            Gradient with respect to the input of the PReLU layer.
        """
        y = np.array(y, dtype="float64")
        dy_dx = np.where(self.last_input > 0, 1.0, self.a)
        self.grad_a = np.where(self.last_input <= 0, y * self.last_input, 0)
        return dy_dx * y

class Linear (Layer):
    """
    Linear layer.
    Default initialization method is Xavier-initialization (well-suited for sigmoid activations).
    Kaiming-initialization available via keyword init='kaiming'.
    'dim_in' and 'dim_out' specify the length of the input and the output vector of the layer.
    """
    def __init__ (self, dim_in:int, dim_out:int, init="xavier"):
        self.name = "linear"
        self.dim_in = dim_in
        self.dim_out = dim_out
        if init == "xavier":
            #xavier formula
            support = (6/(dim_in + dim_out))**0.5
        elif init == "kaiming":
            #kaiming formula
            support = (6/dim_in)**0.5
        else:
            raise ValueError("Incorrect initialization scheme keyword.")
        #initialize weights
        self.weights = np.random.uniform(low=-support, high=support, size=(dim_out, dim_in))
        #initialize biases at zero
        self.bias = np.array([0 for j in range(dim_out)], dtype="float64")
        #slots for memoization
        self.last_input = None
        self.last_output = None
        #slots to store the gradient
        self.weights_grad = np.zeros(shape=(dim_out, dim_in), dtype="float64")
        self.bias_grad = np.array([0 for j in range(dim_out)], dtype="float64")
    def zero_grad (self):
        #reset all gradient slots to zero
        self.weights_grad = np.zeros(shape=(self.dim_out, self.dim_in), dtype="float64")
        self.bias_grad = np.array([0 for j in range(self.dim_out)], dtype="float64")
    def step (self, lr:float):
        self.weights -= lr * self.weights_grad
        self.bias -= lr * self.bias_grad
        #self.bias = self.bias - lr * self.bias_grad
    def forward (self, x):
        x = np.array(x, dtype="float64")
        self.last_input = x
        self.last_output = np.matmul(self.weights, x) +  self.bias
        return self.last_output
    def backward (self, y):
        y = np.array(y, dtype="float64")
        #pass partial gradient backward
        input_grad = np.matmul(y, self.weights)
        #compute gradient for bias vector
        #Recall: d bias = y
        self.bias_grad += y
        #compute gradient for weight matrix
        weights_derivative = np.outer(y, self.last_input)
        self.weights_grad += weights_derivative
        return input_grad

class SGD:
    """
    (Minibatch) stochastic gradient descent.
    """
    def __init__ (self, x_data:list, y_data:list, net:NeuralNet, loss:Loss):
        assert(len(x_data) == len(y_data)), "The lengths of 'x_data' and 'y_data' differ."
        self.x_data = x_data
        self.y_data = y_data
        self.learning_rate = None
        self.gamma = None
        self.loss = loss
        self.losses = [sum([loss.eval(predicted=net.forward(x_data[i]), true=y_data[i]) for i in range(len(x_data))]) / len(x_data)]
        self.net = net
        self.batch_size = None
    def train (self, lr_start=0.1, num_epochs=1, batch_size=float("inf"), gamma=1):
        """
        Train using (minibatch) stochastic gradient descent.
        
        Parameters
        ----------
        lr_start : float, optional
            Initial learning rate. The default is 0.1.
        num_epochs : int, optional
            Number of epochs. The default is 1.
        batch_size : int, optional
            Defaults to "full batch".
        gamma : float, optional
            The decay factor for the exponential learning rate schedule.
            Defaults to '1', a constant learning rate.
        """
        assert(gamma <= 1 and gamma > 0), "'gamma' must be in (0,1]."
        self.learning_rate = lr_start
        self.gamma = gamma
        self.batch_size = min(batch_size, len(self.x_data))
        for e in range(num_epochs):
            indices = list(range(len(self.x_data)))
            #while there are examples that have not yet been used
            while indices:
                #sample a 'batch' of indices of examples not yet used
                batch_indices = sample(indices, min(batch_size, len(indices)))
                #remove the indices of the new batch from the 'reservoir'
                for b in batch_indices:
                    indices.remove(b)
                #zero all gradient slots
                self.net.zero_grad()
                #accumulate the gradients for the batch in the gradient slots
                for i in batch_indices:
                    predicted = self.net.forward(self.x_data[i])
                    d_loss_on_example = self.loss.derivative(self.y_data[i], predicted)
                    self.net.backward(d_loss_on_example)
                #take a step in the direction of the accumulated gradients
                #note that we must scale the gradient with the inverse of the batchsize 
                self.net.step(self.learning_rate * 1/len(batch_indices))
            #compute the loss on the training sample after the additional epoch
            self.losses.append(sum([self.loss.eval(predicted=self.net.forward(self.x_data[i]), true=self.y_data[i]) for i in range(len(self.x_data))]) / len(self.x_data))
            #decay the learning rate before the next epoch
            self.learning_rate *= self.gamma
            print("Epoch "+str(e)+": Loss = ", self.losses[-1])
    def plot_loss (self):
        """
        Plots the loss history.
        """
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim((0,max(self.losses)))
        plt.plot(self.losses, color="black")
        plt.show()
