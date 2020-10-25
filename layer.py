import sys,os
import math
import numpy as np
import scipy as sp
from functions import softmax,Relu,sigmoid

class hiddenLayer:
    def __init__(self,input_size,output_size,activation = 'Relu'):
        self.bias = np.random.randn(output_size)
        self.weight = np.random.randn(output_size,input_size)
        self.which_activation = activation

    def process(self,input):
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
    def __init__(self,size):
        self.size = size

    def process(self,input):
        return input

class outputLayer:
    def __init__(self,size,activation = 'softmax'):
        self.which_activation = activation
        self.size = size
    

    def process(self,input):
        return self.activation(input)
    
    def activation(self,input):
        name = self.which_activation
        if name == 'softmax':
            return softmax(input)
        elif name == 'ReLU':
            return Relu(input)
        elif name == 'sigmoid':
            return sigmoid(input)