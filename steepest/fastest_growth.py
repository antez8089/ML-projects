import cec2017
import numpy as np
from cec2017.functions import f1, f2, f3
from autograd import grad
from time import sleep


def steepest_ascent(f, x, beta, steps=10000, square = 100):
    q = f
    gradient = grad(q)
    x_point = [x]
    for i in range(0, steps):
        x = x - beta * gradient(x)
        for i in range(0,len(x)):
            if x[i] > square:
                x[i] = square
            elif x[i] < -square:
                x[i] = -square
        x_point.append(x)
    return x, x_point, q(x)
