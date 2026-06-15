import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load CSV
df = pd.read_csv(r"C:\Users\nz\Downloads\winedataset\Wine dataset.csv")

# Features and Target
X = df.drop("class", axis=1).values
y = df["class"].values - 1   # classes 1,2,3 -> 0,1,2

# Normalize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Neural Network
class WineNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.network(x)

# Model
model = WineNN()

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 50

for epoch in range(epochs):

    for X_batch, y_batch in train_loader:

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

# Evaluation
model.eval()

with torch.no_grad():

    outputs = model(X_test)

    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == y_test).float().mean()

    print(f"\nAccuracy: {accuracy.item()*100:.2f}%")