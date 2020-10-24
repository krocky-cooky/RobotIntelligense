import numpy as np
import math

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

def sigmoid(x):
    return 1/(1+np.exp(-x))


def ReLU(x):
    if x > 0:
        return x
    else :
        return 0
