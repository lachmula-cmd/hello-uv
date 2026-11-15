import numpy as np
import matplotlib.pyplot as plt
from hello_uv.gd import descend

learning_rate = [0.01,0.1,0.19]
x = np.array([1,2,3])    
y = np.array([3,5,7])
for lr in learning_rate:
    w,b,losses = descend(0,0,x,y,lr,200)
    plt.plot(losses, label=f"lr={lr}")
    print(w,b,losses[-1])

plt.yscale("log")
plt.legend()

plt.savefig("losses_curve.png")