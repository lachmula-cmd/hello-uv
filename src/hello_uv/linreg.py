import numpy as np 


def predict(w,b,x):
    return w*x +b



def mse(w,b,x,y):
    a = (predict(w,b,x) - y) **2 
    
    return np.mean(a)

def gradient(w,b,x,y):
    res = predict(w,b,x) - y
    
    part_w = np.mean(2 * res * x)

    part_b = np.mean(2 * res) 
    
    return part_w,part_b

def numeric_gradient(w,b,x,y,h=1e-5):
    w1 = mse(w+h, b, x, y)
    w2 = mse(w-h, b, x, y)
    b1 = mse(w,b+h,x,y)
    b2 = mse(w,b-h,x,y)
    grad_w = (w1 - w2)/(2 * h)
    grad_b = (b1 - b2)/(2*h)

    return grad_w, grad_b 