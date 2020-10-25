from neural import neuralNetwork
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split



digit = load_digits()
x = digit.data
t = digit.target
t = np.identity(10)[t]
x_train,x_test,t_train,t_test = train_test_split(x,t,random_state=0)

if __name__ == '__main__':
    network = neuralNetwork()
    layer_list = [64,[30,10],10]
    network.set_layer(layer_list)

    for i in range(50000):
        train_size = x_train.shape[0]
        batch_size = 200
        batch_mask = np.random.choice(train_size,batch_size)
        #print(x)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        y = network.calc(x_batch)
        network.backword_propagation(y,t_batch)
        if i%100 == 0:
            print(neuralNetwork.loss(y,t_batch))
    
    y = network.calc(x_test)
    y = np.argmax(y,axis=1)
    t_test = np.argmax(t_test,axis = 1)
    accuracy = np.sum(y == t_test)/float(x_test.shape[0])
    print('accuracy : {}'.format(accuracy))

    #print(network.layers[2].v)