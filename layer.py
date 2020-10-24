import sys,os
import math
import numpy as np
import scipy as sp
from activation import softmax,ReLU,sigmoid

class hiddenLayer:
    def __init__(self,input_size,output_size,activation = 'softmax'):
        self.bias = np.zeros(output_size)
        self.weight = np.zeros((output_size,input_size))
        self.which_activation = activation

    def output(self,input):
        out = np.dot(self.weight,input) + self.bias
        return self.activation(out)

    def activation(self,input):
        name = self.which_activation
        if name == 'softmax':
            return softmax(input)
        elif name == 'ReLU':
            return Relu(input)
        elif name == 'sigmoid':
            return sigmoid(input)


class inputLayer:
    def __init__(self,input_size):
        self.layer = np.zeros(input_size)

class outputLayer:
    def __init__(self,output_size):
        pass
    

    def output(self,input):
        pass