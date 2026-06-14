import torch

x = torch.tensor(5.0, requires_grad=True)

lr = 0.01
epochs = 20

for epoch in range(epochs):

    y = x**4 - 4*x**3 + x**2 + 10

    y.backward()

    with torch.no_grad():
        x -= lr * x.grad

    print("Iteration:", epoch + 1)
    print("x:", x.item())
    print("y:", y.item())
    print("gradient:", x.grad.item())
    print("----------------")

    x.grad.zero_()