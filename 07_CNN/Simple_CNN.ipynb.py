import torch 
import torch.nn as nn

image = torch.randn(1,1,28,28)

conv = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1
)

output = conv(image)
print("input image", image.shape)
print("output image", output.shape)


# example of pooling in cnn 

import torch 
import torch .nn as nn

x = torch.randn(1,8,28,28)

pool = nn.MaxPool2d(kernel_size=2)

y = pool(x)

print("before pooling", x.shape)
print("after pooling", y.shape)