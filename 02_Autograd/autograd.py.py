import torch

x = torch.tensor(1.0, requires_grad=True)

y = x**3 + 2*x**2 + x

y.backward()

print(y.item())
print("gradient", x.grad.item())