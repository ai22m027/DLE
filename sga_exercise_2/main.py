from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from collections import OrderedDict
from typing import List, Tuple
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader

import flwr as fl
from flwr.common import Metrics

NUM_CLIENTS = 3
categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "sex", "native-country", "salary-class"]
numeric_cols = ["age", "capital-gain", "hours-per-week"]

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # Assuming 2 classes for salary-class

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class AdultDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].to_dict()
        if self.transform:
            sample = self.transform(sample)
        return sample

class SingleTableMetadata:
    def __init__(self):
        pass  # Implement as needed

    def detect_from_csv(self, filepath):
        pass  # Implement as needed

    def visualize(self):
        pass  # Implement as needed

def load_adult_dataset():
    # Load adult data
    data_path = "sga_data/ex1/adult-subset-for-synthetic.csv"
    metadata_table = SingleTableMetadata()
    original_data = pd.read_csv(data_path)
    metadata_table.detect_from_csv(filepath=data_path)
    metadata_table.visualize()

    categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "sex", "native-country", "salary-class"]
    numeric_cols = ["age", "capital-gain", "hours-per-week"]

    # Combine categorical and numeric columns
    all_cols = numeric_cols + ["salary-class"]

    # Drop categorical columns
    original_data = original_data[all_cols]

    # Define a simple transformation function (you can customize this based on your needs)
    def transform(sample):
        # Normalize numeric columns
        for col in numeric_cols:
            sample[col] = (sample[col] - original_data[col].mean()) / original_data[col].std()

        # Convert to PyTorch tensors
        features = torch.tensor([sample[col] for col in numeric_cols], dtype=torch.float32)
        label_mapping = {'<=50K': 0, '>50K': 1}
        label = torch.tensor(label_mapping[sample["salary-class"]], dtype=torch.long)

        return {"features": features, "label": label}

    # Create PyTorch dataset
    dataset = AdultDataset(original_data, transform=transform)

    # Split the dataset into train/val/test loaders
    fraction_size = len(dataset) // 4
    training_size = int(fraction_size * 0.8)
    val_size = fraction_size - training_size
    test_size = len(dataset) - 3 * fraction_size

    tr_1, val_1, tr_2, val_2, tr_3, val_3, test = torch.utils.data.random_split(dataset, [training_size, val_size, training_size, val_size, training_size, val_size, test_size])

    train_loaders = [DataLoader(tr_1), DataLoader(tr_2), DataLoader(tr_3)]
    val_loaders = [DataLoader(val_1), DataLoader(val_2), DataLoader(val_3)]
    test_loader = DataLoader(test)

    return train_loaders, val_loaders, test_loader

def train(model, train_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            features, labels = batch['features'].to(device), batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            features, labels = batch['features'].to(device), batch['label'].to(device)

            outputs = model(features)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    average_loss = loss / len(test_loader)

    return average_loss, accuracy

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Load the data
train_loaders, val_loaders, test_loader = load_adult_dataset()

# Training loop, only on a single client
"""
for client_id, train_loader in enumerate(train_loaders):
    print(f"Training Client {client_id + 1}")
    
    # Define and initialize the neural network
    net = Net(input_size=len(numeric_cols)).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training
    for epoch in range(5):
        train(net, train_loader, 1, "cuda" if torch.cuda.is_available() else "cpu")
        loss, accuracy = test(net, val_loaders[client_id], "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

# Final test set performance
loss, accuracy = test(net, test_loader, "cuda" if torch.cuda.is_available() else "cpu")
print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
"""
        
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1, device= "cuda" if torch.cuda.is_available() else "cpu")
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, "cuda" if torch.cuda.is_available() else "cpu")
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net(input_size=len(numeric_cols)).to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = train_loaders[int(cid)]
    valloader = val_loaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=2,  # Never sample less than 10 clients for training
    min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
    min_available_clients=3,  # Wait until all 10 clients are available
)

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=2,
    min_evaluate_clients=1,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)

pass