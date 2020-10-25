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


def Relu(x):
    if x > 0:
        return x
    else :
        return 0


def euler_loss(y,t):
    loss = np.sum((y-t)**2)/2

    return loss

