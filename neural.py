from layer import hiddenLayer,inputLayer,outputLayer
from functions import euler_loss
import numpy as np


class neuralNetwork:
    def __init__(self,learning_rate = 0.0001,layer_list = [],iters_num = 10000):
        self.layers = list()
        self.deltas = list()
        self.learning_rate = learning_rate
        self.iters_num = 10000
        if layer_list:
            self.set_layer(layer_list)

    def set_layer(self,layer_list):
        input_size = layer_list[0]
        output_size = layer_list[2]
        hidden_layers = layer_list[1] 
        
        input_layer = inputLayer(input_size)
        self.layers.append(input_layer)
        former = input_size
        for sz in hidden_layers:
            layer = hiddenLayer(input_size=former,output_size=sz,learning_rate=self.learning_rate,activation='Relu')
            self.layers.append(layer)
            former = sz
            #delta = np.zeros(layer_list[i])
            #self.deltas.append(delta)

        output_layer = outputLayer(input_size=former,output_size=output_size,activation='identity',learning_rate=self.learning_rate)
        self.layers.append(output_layer)
        print('successfully layers are updated')
    
    

    def calc(self,input):
        
        vector = input
        for x in self.layers:
            vector = x.process(vector)

        return vector

    @classmethod
    def loss(self,y,t):
        loss = euler_loss(y,t)
        return loss

    def backword_propagation(self,y,t):
        dif = y-t
        layers = self.layers[1:]
        for layer in reversed(layers):
            layer.update_delta(dif)
            dif = layer.send_backword()
            layer.update_weight()


    def train(self,train,test):
        pass

