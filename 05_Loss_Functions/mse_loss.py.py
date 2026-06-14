# mse_loss.py
import torch
import torch.nn as nn

y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])
y_pred = torch.tensor([1.5, 4.5, 5.5, 8.0])

loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_true)

print("MSE Loss:", loss.item())