from hello_uv.linreg import mse, gradient


def step(w,b,x,y,lr):
    g = gradient(w,b,x,y)
    w2,b2 = g
    new_w = w-lr*w2
    new_b = b - lr*b2
    return new_w,new_b

def descend(w,b,x,y,lr,steps):
    losses = []
    losses.append(mse(w,b,x,y))
    for _ in range(steps):
        w,b = step(w,b,x,y,lr)
        losses.append(mse(w,b,x,y))
    return w,b,losses
