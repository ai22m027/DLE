import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

#function R^2 -> R
def function (x1,x2):
    return x1*x1 + 0.25*x2*x2
def gradient (x1,x2):
    return 2*x1, 0.5*x2

def plot_function (function:callable, minimum=-5, maximum=5, mesh=100):
    Function = np.vectorize(function)
    x = np.linspace(minimum, maximum, mesh)
    y = np.linspace(minimum, maximum, mesh)
    X,Y = np.meshgrid(x, y)
    Z = Function(X,Y)
    ax = plt.axes(projection='3d')
    #ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, color="black")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.gray, edgecolor='none')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')
    plt.show()

def plot_field_trajectory (field:callable, minimum=-5, maximum=5, mesh=20, scaling=0.05, startpoint:tuple=None, stepsize=0.1, num_steps=10):
    """
    Plot the negative gradient field of the function.
    If a startpoint is given the respective trajectory of gradient descent is also plotted.

    Parameters
    ----------
    field : callable
        The gradient function.
    minimum : float, optional
        Minimum for the bounding box for plotting. The default is -5.
    maximum : float, optional
        Maximum for the bounding box for plotting. The default is 5.
    mesh : int, optional
        Controls number of points at which vectors will be plotted (mesh^2). The default is 20.
    scaling : float, optional
        Positive number to scale the plotted vectors of the field for better visibility. The default is 0.05.
    startpoint : tuple, optional
        Startpoint for a trajectory. The default is None.
    stepsize : float, optional
        Positive stepsize for gradient descent. The default is 0.1.
    num_steps : TYPE, optional
        Number of steps of gradient descent. The default is 10.
    """
    if startpoint is not None:
        color = "gray"
    else:
        color = "black"
    ax = plt.axes()
    ax.set_aspect("equal")
    ax.set_title("Negative gradient field of f(x1,x2)")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    x = np.linspace(minimum, maximum, mesh)
    y = np.linspace(minimum, maximum, mesh)
    for xx in x:
        for yy in y:
            grad = gradient(xx,yy)
            ax.arrow(xx,yy, -scaling*grad[0],-scaling*grad[1], head_width=0.05, width=0.0075, color=color)
    if startpoint is not None:
        for i in range(num_steps):
            grad = gradient(startpoint[0], startpoint[1])
            ax.arrow(startpoint[0],startpoint[1], -stepsize*grad[0],-stepsize*grad[1], head_width=0.07, width=0.0075, color="black")
            startpoint = (startpoint[0]-stepsize*grad[0], startpoint[1]-stepsize*grad[1])
    plt.show()

#plot graph of function
plot_function(function)

#plotting negative gradient field
plot_field_trajectory(gradient, mesh=25, scaling=0.04)

#As computed on the slides
plot_field_trajectory(gradient, mesh=25, scaling=0.04, startpoint=(2,4), num_steps=1, stepsize=0.2)

#plotting a trajectory of gradient descent
plot_field_trajectory(gradient, mesh=25, scaling=0.04, startpoint=(2,4), num_steps=30, stepsize=0.2)

#learning rate too high
plot_field_trajectory(gradient, mesh=25, scaling=0.04, startpoint=(1,1), num_steps=30, stepsize=1)

#learning rate too low
plot_field_trajectory(gradient, mesh=25, scaling=0.04, startpoint=(2,4), num_steps=30, stepsize=0.01)

pass







