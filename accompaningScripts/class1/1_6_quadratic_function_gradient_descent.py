import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from math import sin, cos

class GradientDescent (object):
    """
    Gradient descent in the one-dimensional case.
    For funtions R -> R.
    """
    def __init__ (self, function:callable, derivative:callable, x:float):
        """
        Parameters
        ----------
        function : callable
            A function R -> R to enact gradient descent upon.
        derivative : callable
            The derivative of 'function'.
        x : float
            Initial x-value for gradient descent.
        """
        self.function = function
        self.derivative = derivative
        self.x = x
        self.history = [x]
    def step (self, lr:float):
        assert(lr > 0), "The learning rate must be geater zero."
        #step in opposite direction of derivative
        self.x -= lr*self.derivative(self.x)
        self.history.append(self.x)
    def gd (self, lr:float, num:int=100):
        """
        Enact 'num' steps of gradient descent with learning rate 'lr'.
        """
        for i in range(num):
            self.step(lr)
    def plot_function (self, title=""):
        """
        Plot the function to standard output.
        """
        plt.title(title)
        grid = np.linspace(min(self.history)-10, max(self.history)+10, 1000)
        plt.plot(grid, [self.function(x) for x in grid], color="black")
        plt.show()
    def plot_values (self, title=""):
        """
        Plot the history of values to standard output.
        """
        plt.title(title)
        plt.plot([self.function(x) for x in self.history], color="black")
        plt.show()
    def plot_history (self, title=""):
        """
        Visualize the historic dynamic of gradient descent.
        """
        plt.title(title)
        grid = np.linspace(min(self.history)-1, max(self.history)+1, 1000)
        plt.plot(grid, [self.function(x) for x in grid], color="black")
        plt.plot(self.history, [self.function(x) for x in self.history], ls="dashed", color="gray")
        plt.scatter(self.history, [self.function(x) for x in self.history], cmap=cm.magma, c=np.linspace(100,0,len(self.history)), zorder=3, s=15)
        plt.show()

#squaring function
#divergence for lr > 1
#convergence for lr < 1
squaring = GradientDescent(function=lambda x: x*x, derivative=lambda x: 2*x, x=0.8)
squaring.plot_function(title="Squaring function")
squaring.gd(0.01, 10)
squaring.plot_history(title="Squaring function point history")
squaring.plot_values(title="Squaring function value history")

#local minima function
#small differences in starting value
#(e.g. slightly to the left of a local maximum versus slightly to the right of it)
#determine the resting point
local_minima = GradientDescent(function=lambda x: 2*x + sin(5*x), derivative=lambda x: cos(5*x)*5 + 2, x=1.7)
local_minima.plot_function(title="Local minima function")
local_minima.gd(0.02)
local_minima.plot_history(title="Local minima point history")
local_minima.plot_values(title="Local minima value history")

#oscillating function
#from the left of the function a global optimum is reached (e.g. x < -1)
#from the right of the graph convergence is not reached (e.g. x > 1)
#in the oscillatory region behavior appears chaotic (e.g. x = 0.5)
oscillating = GradientDescent(function=lambda x: sin(1/x), derivative=lambda x: cos(1/x) * (-1/x**2), x=-1.1)
oscillating.plot_function(title="Oscillating function")
oscillating.gd(0.03)
oscillating.plot_history(title="Oscillating function point history")
oscillating.plot_values(title="Oscillating function value history")

