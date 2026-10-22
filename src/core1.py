import math

def f(x):
    return 2 *x **2 -1

def total(xs):
    x= 0
    for i in xs:
        x += i
    return x

def product(xs):
    x = 1
    for i in xs:
        x = x * i
    return x

def mean(xs):
    z = len(xs)
    t = total(xs)
    return t / z

def sum_log(xs):
    z = []
    for i in xs:
        z.append(math.log(i))
    z = total(z)
    return z

if __name__ == "__main__":
    print("core1 loaded")