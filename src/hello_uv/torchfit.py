from torch import nn
import torch

class LinReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1,out_features=1, dtype=torch.float64)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self,x):
        return self.linear(x)

def fit(x,y,lr,steps,opt="sgd"):
    model = LinReg()

    x = torch.tensor(x, dtype=torch.float64).reshape(-1,1)
    y = torch.tensor(y, dtype=torch.float64).reshape(-1,1)

    if opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        forward = model(x)
        error_sqr = (forward - y)**2
        loss = error_sqr.mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    w = model.linear.weight
    b = model.linear.bias

    return w.item(), b.item(), losses
         




