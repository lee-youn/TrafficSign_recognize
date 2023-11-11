import numpy as np
import torch

a = np.random.rand(10,3,48,48)
print(a.shape)
a = a.reshape(10,-1)
print(a.shape)

b = torch.from_numpy(a)
print(b.shape)
b = b.reshape(10,-1)
print(b.shape)

var = np.mean(a, axis=0)
print(var.shape)
var2 = torch.mean(b,dim=0)
print(var2.shape)

print(np.sqrt(var + 10e-7))
print(torch.sqrt(var2 + 10e-7))

#var = torch.mean(xc**2, dim=(N,D),keepdim=False)  # np.mean(xc**2, axis=0)
#std = torch.sqrt(var + 10e-7)                     # np.sqrt(var + 10e-7)