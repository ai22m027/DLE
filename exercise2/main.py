import numpy as np
import matplotlib.pyplot as plt
from random import sample
from candle import *

# Create a dataset for XOR
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float64")
y_data = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype="float64")

# Define and construct the neural network
net = NeuralNet()
net.add_layer(Linear(2, 10))
net.add_layer(PReLU(10))
for _ in range(20):
    net.add_layer(Linear(10, 10))  # Assuming you have a Dense layer implementation
    net.add_layer(PReLU(10))
    
net.add_layer(Linear(10, 2))
net.add_layer(Logistic())

# Define the loss function (e.g., MSE loss)
loss = SELoss()

# Create an SGD instance for training
sgd = SGD(x_data.tolist(), y_data.tolist(), net, loss)

# Train the neural network
sgd.train(lr_start=0.01, num_epochs=5000, gamma=1)

# Plot the loss history
sgd.plot_loss()

# Test the network on XOR inputs
for i in range(len(x_data)):
    x = x_data[i]
    predicted = net.forward(x)
    print(f"Input: {x}, Predicted: {predicted}")