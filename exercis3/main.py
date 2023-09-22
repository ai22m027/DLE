import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from ResnetsMasked import archimedean_spiral

# Define the residual block class
class ResidualBlock(nn.Module):
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

# Create the neural network using the residual blocks
class ResidualNet(nn.Module):
    def __init__(self, num_blocks):
        super(ResidualNet, self).__init__()
        self.blocks = nn.ModuleList([ResidualBlock() for _ in range(num_blocks)])
        self.final_fc = nn.Linear(2, 1)  # Change output to a single unit

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_fc(x)
        return torch.sigmoid(x)  # Apply sigmoid activation to the output

# Data generation and preprocessing
x_data, y_data = archimedean_spiral()
x_data = torch.tensor(x_data, dtype=torch.float32)
y_data = torch.tensor(y_data, dtype=torch.float32)

# Split data into training and test sets
split_ratio = 0.8
split_idx = int(len(x_data) * split_ratio)
x_train, x_test = x_data[:split_idx], x_data[split_idx:]
y_train, y_test = y_data[:split_idx], y_data[split_idx:]

y_train = y_train.float()
y_test = y_test.float()

# Create DataLoader for training and test data
batch_size = 32
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the model, loss function, and optimizer
model = ResidualNet(num_blocks=5)
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss without sigmoid
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # Use squeeze() to remove the extra dimension
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()  # Use squeeze() to remove the extra dimension
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        test_losses.append(test_loss / len(test_loader))

    print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_losses[-1]:.4f} Test Loss: {test_losses[-1]:.4f}')

# Plot the loss evolution
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()