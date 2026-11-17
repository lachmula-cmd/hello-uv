import torch
from hello_uv.linreg import predict



def autograd_gradient(w,b,x,y):
    w = torch.tensor(w,requires_grad=True, dtype=torch.float64)
    b = torch.tensor(b,requires_grad=True, dtype=torch.float64)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    pred = predict(w,b,x)
    sqr_error = (pred - y)**2
    L = sqr_error.mean()

    L.backward()
    w_grad = w.grad
    b_grad = b.grad

    return w_grad.detach().item(),  b_grad.detach().item()






