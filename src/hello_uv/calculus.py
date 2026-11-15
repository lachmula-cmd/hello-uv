import numpy as np
import matplotlib.pyplot as plt

def forward_diff(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

def slope_curve(f, xs, h=1e-5):
    return central_diff(f, xs, h=1e-5)

def flattest_x(xs, slopes):
    slope = np.argmin(np.abs(slopes))
    return xs[slope]