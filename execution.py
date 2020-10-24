from neural import neuralNetwork
import numpy as np

if __name__ == '__main__':
    network = neuralNetwork()
    layer_list = [5,3,6,7,10,2]
    network.set_layer(layer_list)
    x = np.random.rand(5)
    print('input : ' + str(x))
    y = network.calc(x)
    print(y)
