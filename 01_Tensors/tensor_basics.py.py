import torch

# Tensor Creation
x = torch.tensor([1, 2, 3])
print(x)

# Shape
print(x.shape)

# Dtype
print(x.dtype)

# Tensor Operations
y = torch.tensor([4, 5, 6])
print(x + y)
print(x * y)