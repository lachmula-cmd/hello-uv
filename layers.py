import numpy as np

class Module():
    def __init__(self):
        self.training = True

    def __call__(self, x):
        return self.forward(x)

    def num_params(self):
        return 0

    def eval(self):
        self.training = False


class Linear(Module):
    def __init__(self, W, b):
        super().__init__()
        self.W = np.array(W, dtype=float)
        self.b = np.array(b, dtype=float)
        

    def forward(self, x):
        return self.W @ x + self.b

    def num_params(self):
        return self.W.shape[0] * self.W.shape[1] + len(self.b)

class ReLU(Module):
    def forward(self , x):
        return np.maximum(x,0)

class Sequential(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    
    def forward(self, x):
        for layers in self.layers:
            x = layers(x)
        return x 
    
    def num_params(self):
        total = 0
        for layer in self.layers:
            total += layer.num_params()
        return total

    def eval(self):
        super().eval()
        for layer in self.layers:
            layer.eval()
        