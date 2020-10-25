from layer import hiddenLayer,inputLayer,outputLayer
from functions import euler_loss
import numpy as np


class neuralNetwork:
    def __init__(self,learning_rate = 0.001,layer_list = []):
        self.layers = list()
        self.learning_rate = learning_rate
        if layer_list:
            self.set_layer(layer_list)

    def set_layer(self,layer_list):
        
        input_layer = inputLayer(layer_list[0])
        self.layers.append(input_layer)
        for i in range(1,len(layer_list)):
            hidden_layer = hiddenLayer(input_size = layer_list[i-1],output_size = layer_list[i])
            self.layers.append(hidden_layer)
        output_layer = outputLayer(layer_list[len(layer_list)-1])
        self.layers.append(output_layer)
        print('successfully layers are updated')
    

    def calc(self,input):
        if self.layers[0].size != input.shape[0]:
            print('The size of inserted vector is not proper')
            return
        vector = input
        for x in self.layers:
            vector = x.process(vector)

        return vector

    def loss(self,input,t):
        y = self.calc(input)
        loss = euler_loss(y,t)
        return loss
