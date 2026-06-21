import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Example Data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Dataset
dataset = MyDataset(X, y)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# Check batches
for inputs, labels in dataloader:
    print("Inputs:", inputs)
    print("Labels:", labels)
    print()