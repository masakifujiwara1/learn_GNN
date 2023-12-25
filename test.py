import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt
import torch
from torch import nn

num_node = 4
feature_num = 4
out_channels = 8

X = np.random.randn(num_node, feature_num) + 1

A = np.zeros((num_node, num_node))
A += 1
# print(A)

W_tensor = nn.Parameter(torch.randn((feature_num, out_channels)))

def get_D(A, pow=-0.5):
    d_ii = np.sum(A, 0)
    D = np.zeros_like(A)
    for i in range(len(A)):
        D[i, i] = d_ii[i]**(pow)
    return D

D = get_D(A)

# x = torch.from_numpy(x.astype(np.float32)).clone()

D_tensor = torch.tensor(D, dtype=torch.float32)
A_tensor = torch.tensor(A, dtype=torch.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)

XW = X_tensor @ W_tensor
new_X = D_tensor @ A_tensor @ D_tensor @ XW

print(new_X.size())
print(new_X)
