from neural import neuralNetwork
import numpy as np
from sklearn.datasets import load_digits

digit = load_digits()
x = digit.data
t = digit.target
t = np.identity(10)[t]

if __name__ == '__main__':
    network = neuralNetwork()
    layer_list = [64,[30],10]
    network.set_layer(layer_list)

    for i in range(2):
        #print(x)
        y = network.calc(x)
        network.backword_propagation(y,t)
        print(neuralNetwork.loss(y,t))
    #print(network.layers[2].v)