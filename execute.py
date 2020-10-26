from neural import neuralNetwork
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import datetime



digit = load_digits()
x = digit.data
t = digit.target
t = np.identity(10)[t]
x_train,x_test,t_train,t_test = train_test_split(x,t,random_state=0)

if __name__ == '__main__':
    network = neuralNetwork(epoch=50000)
    layer_list = [64,[30,40,30],10]
    network.set_layer(layer_list)
    network.train(x_train,t_train)
    y_test = network.predict(x_test)
    loss = neuralNetwork.loss(y_test,t_test)
    accuracy = network.accuracy(x_test,t_test)
    print('loss : {}'.format(loss))
    print('accuracy : {}'.format(accuracy))


    path = './results.txt'
    with open(path) as f:
        text = f.read()

    text += '=========' + str(datetime.datetime.now()) + '=========\n'
    text += 'layer_list : ' + str(layer_list) + '\n'
    text += 'learning_rate : ' + str(network.learning_rate) + '\n'
    text += 'epoch : ' + str(network.epoch) + '\n'
    text += 'batch_per : ' + str(network.batch_per) + '\n'
    text += 'loss_list : ' + str(network.loss_list[::2]) + '\n'
    text += 'acc_list : ' + str(network.acc_list[::2]) + '\n'
    text += 'test_loss : ' + str(loss) + '\n'
    text += 'test_accuracy : ' + str(accuracy) + '\n'
    text += '\n\n'
    with open('./results.txt',mode='w') as f:
        f.write(text)


    #print(network.layers[2].v)