from math import sin, cos
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

#data generation

def archimedean_spiral (maximum:float=75, num:int=100, noise:float=0):
    """
    Sampling data from a noisy Archimedean spiral.

    Parameters
    ----------
    maximum : float, optional
        Non-negative number that determines how much of the spiral is used. The default is 75.
    num : integer, optional
        The number of sampled points. The default is 100.
    noise : float, optional
        Non-negative number determining the range of uniform noise. The default is 0.

    Returns
    -------
    x_data :
        Feature vector.
    y_data :
        Target vector.
    """
    #for approximately even spacing
    grid = [k**0.5 for k in np.linspace(1,maximum,int(num/2))]
    #formula for Archimedean spiral
    x = [t*cos(t)for t in grid]
    y = [t*sin(t) for t in grid]
    #multiply coordinates by -1 for second class
    x2 = [-xx for xx in x]
    y2 = [-yy for yy in y]
    #add noise
    x = [xx + noise * np.random.normal() for xx in x]
    y = [yy + noise * np.random.normal() for yy in y]
    x2 = [xx + noise * np.random.normal() for xx in x2]
    y2 = [yy + noise * np.random.normal() for yy in y2]
    #create array for the features
    class0 = [list(tup) for tup in zip(x,y)]
    class1 = [list(tup) for tup in zip(x2,y2)]
    x_data = class0 + class1
    #create binary target
    y_data = [0 for _ in range(len(class0))] + [1 for _ in range(len(class1))]
    return x_data, y_data

def plot_classes (x_data, y_data, title=""):
    """
    Convenience function to plot data.
    """
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.set_title(title)
    ax.scatter([x_data[i][0] for i in range(len(x_data)) if y_data[i]==0],
                [x_data[i][1] for i in range(len(x_data)) if y_data[i]==0],
                s=10)
    ax.scatter([x_data[i][0] for i in range(len(x_data)) if y_data[i]==1],
                [x_data[i][1] for i in range(len(x_data)) if y_data[i]==1],
                s=10)
    fig.show()
    plt.show()
    
    
class ResidualBlock(nn.Module):
    """Residual block according to provided PDF.

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out += residual  # Skip connection
        return out


class ResidualNet(nn.Module):
    """ResidualNet class. Consists of a series of ResidualBlock(s) 
    and a Linear block in the end with sigmoid activation function.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_blocks):
        super(ResidualNet, self).__init__()
        self.blocks = nn.ModuleList([ResidualBlock() for _ in range(num_blocks)])
        self.final_fc = nn.Linear(2, 1) # lin output layer

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_fc(x)
        return torch.sigmoid(x)  # sigmoid output, cause why not
    
def plot_intermediate_representations(model, x_data, y_data, title=""):
    """Helper function to plot interm. results of final model.

    Args:
        model (_type_): ResNet model
        x_data (_type_): training data
        y_data (_type_): training data labels
        title (str, optional): Plot title. Defaults to "".
    """
    model.eval()
    with torch.no_grad():
        x_data = torch.tensor(x_data, dtype=torch.float32)
        intermediate_outputs = []

        def hook(module, input, output):
            intermediate_outputs.append(output)

        hooks = []
        for block in model.blocks:
            hook_handle = block.register_forward_hook(hook)
            hooks.append(hook_handle)

        model(x_data)

        num_blocks = len(intermediate_outputs)
        fig, axes = plt.subplots(1, num_blocks, figsize=(15, 3))
        fig.suptitle(title)

        for i in range(num_blocks):
            intermediate_data = intermediate_outputs[i].numpy()
            class0_indices = [j for j in range(len(x_data)) if y_data[j] == 0]
            class1_indices = [j for j in range(len(x_data)) if y_data[j] == 1]

            axes[i].scatter(intermediate_data[class0_indices, 0], intermediate_data[class0_indices, 1], s=10, label='Class 0', marker='o')
            axes[i].scatter(intermediate_data[class1_indices, 0], intermediate_data[class1_indices, 1], s=10, label='Class 1', marker='x')
            axes[i].set_title(f'Block {i + 1}')
            axes[i].legend()

        plt.show()

        for hook_handle in hooks:
            hook_handle.remove()

def main():
    #gen data
    x_data, y_data = archimedean_spiral(num=200, noise=0.2, maximum=75)
    plot_classes(x_data, y_data, title="Generated data")

    x_data = torch.tensor(x_data, dtype=torch.float32)
    y_data = torch.tensor(y_data, dtype=torch.float32)

    # train - test split
    split_ratio = 0.8
    split_idx = int(len(x_data) * split_ratio)
    x_train, x_test = x_data[:split_idx], x_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]

    y_train = y_train.float()
    y_test = y_test.float()

    # data loader, required for tensor
    batch_size = 32
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # model, lossfun, optimizer
    model = ResidualNet(num_blocks=5)
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss without sigmoid
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 1000 # sometimes it converges nicely, sometimes it needs more epochs
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            test_losses.append(test_loss / len(test_loader))

        print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_losses[-1]:.4f} Test Loss: {test_losses[-1]:.4f}')

    '''
    The model sometimes can't converge. Please just rerun the script. Could runs lead to a loss of ~ 0.3 after 1000 epochs.
    '''

    # Task 1
    # Plot the loss evolution
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Task 2
    # Plot the intermediat state of the ResNet
    plot_intermediate_representations(model, x_data, y_data, title="Intermediate Representations of Final Model")

if __name__ == "__main__":
    main()
