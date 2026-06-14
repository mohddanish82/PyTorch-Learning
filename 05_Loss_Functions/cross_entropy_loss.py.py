# cross_entropy_loss.py
import torch
import torch.nn as nn

# 3 classes example
y_pred = torch.tensor([[2.0, 1.0, 0.1]])  # logits
y_true = torch.tensor([0])  # correct class index

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(y_pred, y_true)

print("Cross Entropy Loss:", loss.item())