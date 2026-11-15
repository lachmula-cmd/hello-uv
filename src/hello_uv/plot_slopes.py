import matplotlib.pyplot as plt
import numpy as np
from hello_uv.calculus import slope_curve
from hello_uv.calculus import flattest_x
x = np.linspace(-2.5, 2.5, 501)

def f(x):
    return pow(x, 3) - 3 * x


f1 = f(x)
f2 = slope_curve(f, x)
f3 = 3*pow(x, 2) - 3
xs1 = x[x > 0]
fxs1 = f2[x > 0]
xs2 = x[x<0]
fxs2 = f2[x<0]
pointx1 = flattest_x(xs1, fxs1)
pointx2 = flattest_x(xs2, fxs2)
pointy1 = f(pointx1)
pointy2 = f(pointx2)
plt.plot(x,f1, label="f1")
plt.plot(x, f2, label="f2")
plt.plot(x,f3, "--", label="f3")
plt.scatter([pointx1, pointx2], [pointy1, pointy2])
plt.title("slopes plot")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.axhline(0)
plt.legend()
plt.savefig("slopes1.png")
plt.close()

