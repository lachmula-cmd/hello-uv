import numpy as np

a = np.arange(6).reshape(2, 3)
b = a.reshape(-1)
c = np.linalg.norm(a, keepdims=True)
d = np.argsort(b)[::-1]

print(a, b.shape, c, c.shape, d)