from layer import hiddenLayer,inputLayer,outputLayer
from functions import euler_loss
import numpy as np


class neuralNetwork:
    def __init__(self,learning_rate = 0.0001,epoch = 20000,batch_per = 0.6):
        self.layers = list()
        self.deltas = list()
        self.learning_rate = learning_rate
        self.batch_per = batch_per
        self.epoch = epoch
        self.loss_list = list()
        self.acc_list = list()
        

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
    
    

    def predict(self,input):
        
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


    def train(self,x,t):
        train_size = x.shape[0]
        batch_size = int(train_size*self.batch_per)
        for i in range(self.epoch):
            batch = np.random.choice(train_size,batch_size)
            x_batch = x[batch]
            t_batch = t[batch]
            y = self.predict(x_batch)
            if i%100 == 0:
                word = '========epoch' + str(i+1) + '========='
                print(word)
                loss = neuralNetwork.loss(y,t_batch)
                y_sub = np.argmax(y,axis=1)
                t_sub = np.argmax(t_batch,axis=1)
                acc = np.sum(y_sub == t_sub)/float(batch_size)
                self.loss_list.append(loss)
                self.acc_list.append(acc)
                print('loss : ' + str(loss))
                print('accuracy : ' + str(acc))
                word = '='*len(word)
                print(word)

            self.backword_propagation(y,t_batch)
            
    def accuracy(self,x,t):
        y = self.predict(x)
        y_sub = np.argmax(y,axis=1)
        t_sub = np.argmax(t,axis=1)
        acc = np.sum(y_sub == t_sub)/float(y.shape[0])
        return acc
        

        

