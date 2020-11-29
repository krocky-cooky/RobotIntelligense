import sys,os
import math
import numpy as np
import scipy as sp
from functions import softmax,Relu,sigmoid,identity

class hiddenLayer:
    def __init__(
        self,
        input_size,
        output_size,
        activation = 'Relu',
        learning_rate=0.001,
        optimize_initial_weight = True
    ):
        self.bias = np.zeros((1,output_size))
        self.weight = None
        if optimize_initial_weight:
            self.weight = np.random.randn(input_size,output_size)/math.sqrt(input_size)
        else:
            self.weight = 0.01*np.random.randn(input_size,output_size)
        self.which_activation = activation
        self.learning_rate = learning_rate
        self.v = None
        self.y = None
        self.delta = None
        self.input = None

    def process(self,input):
        self.input = input
        self.v = np.dot(input,self.weight) + self.bias
        self.y = self.activation(self.v)
        return self.y

    def activation(self,input,div=False):
        name = self.which_activation
        if name == 'softmax':
            return softmax(input,div)
        elif name == 'Relu':
            return Relu(input,div)
        elif name == 'sigmoid':
            return sigmoid(input,div)
        elif name == 'identity':
            return identity(input,div)

    def update_delta(self,dif):
        self.delta = self.activation(self.v,div=True)*dif
        


    def update_weight(self):
        x = self.learning_rate/self.delta.shape[0]*np.dot(self.input.T,self.delta)
        self.weight -= x
        self.bias -= self.learning_rate/self.delta.shape[0]*np.sum(self.delta,axis=0)




    def send_backword(self):
        dif = np.dot(self.delta,self.weight.T)
        return dif




class inputLayer:
    def __init__(self,size):
        self.size = size

    def process(self,input):
        return input



class outputLayer:
    def __init__(
        self,
        input_size,
        output_size,
        activation = 'identity',
        learning_rate=0.001,
        optimize_initial_weight = True
    ):
        self.bias = np.zeros((1,output_size))
        self.weight = None
        if optimize_initial_weight:
            self.weight = np.random.randn(input_size,output_size)/math.sqrt(input_size)
        else:
            self.weight = 0.01*np.random.randn(input_size,output_size)

        self.which_activation = activation
        self.learning_rate = learning_rate

    def process(self,input):
        self.input = input
        self.v = np.dot(input,self.weight) + self.bias
        self.y = self.activation(self.v)
        return self.y

    def activation(self,input,div = False):
        name = self.which_activation
        if name == 'softmax':
            return softmax(input,div)
        elif name == 'Relu':
            return Relu(input,div)
        elif name == 'sigmoid':
            return sigmoid(input,div)
        elif name == 'identity':
            return identity(input,div)

    def update_delta(self,dif):
        self.delta = self.activation(self.v,div=True)*dif

    def update_weight(self):
        self.weight -= self.learning_rate/self.delta.shape[0]*np.dot(self.input.T,self.delta)
        self.bias -= self.learning_rate/self.delta.shape[0]*np.sum(self.delta,axis=0)


    def send_backword(self):
        dif = np.dot(self.delta,self.weight.T)
        return dif