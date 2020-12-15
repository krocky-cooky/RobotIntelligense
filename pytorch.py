import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.optim as optimizers

class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim_1,
        hidden_dim_2,
        output_dim
    ):
        super().__init__()
        self.l1 = nn.Linear(input_dim,hidden_dim_1)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim_1,hidden_dim_2)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_dim_2,output_dim)
        self.a3 = nn.Softmax(dim = 1)
        
        self.layers = [self.l1,self.a1,self.l2,self.a2,self.l3,self.a3]
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            
        return x

class Dataset:
    def __init__(self,train = True):
        x_train = np.load('./datasets/kmnist-train-imgs.npz')['arr_0']
        t_train = np.load('./datasets/kmnist-train-labels.npz')['arr_0']
        x_test = np.load('./datasets/kmnist-test-imgs.npz')['arr_0']
        t_test = np.load('./datasets/kmnist-test-labels.npz')['arr_0']

        t_train = np.identity(10)[t_train]
        t_test = np.identity(10)[t_test]
        x_train = x_train.reshape((60000,-1)).astype(float)
        x_test = x_test.reshape((10000,-1)).astype(float)
        x_train = x_train/255
        x_test = x_test/255
        if train:
            #self.x = torch.from_numpy(x_train).float
            #self.t = torch.from_numpy(t_train).long
            self.x = x_train
            self.t = t_train
            self.size = x_train.shape[0]
        else:
            #self.x = torch.from_numpy(x_test).float
            #self.t = torch.from_numpy(t_test).long
            self.x = x_test
            self.t = t_test
            self.size = x_test.shape[0]
    def __len__(self):
        return self.size
    
    def __getitem__(self,index):
        return self.x[index],self.t[index]

def load_kuzushijiMNIST(batch_size = 500):
    train_loader = torch.utils.data.DataLoader(
        Dataset(True),
        batch_size = batch_size,
        shuffle = True,
    )
    test_loader = torch.utils.data.DataLoader(
        Dataset(False),
        batch_size = batch_size,
        shuffle = True,
    )
    
    return {
        'train' : train_loader,
        'test' : test_loader
    }


if __name__ == '__main__':
    epoch = 5
    history = {
        'train_loss' : [],
        'test_loss' : [],
        'test_acc' : []
    }

    net = Net(784,400,40,10)
    loaders = load_kuzushijiMNIST()
    

    optimizer = optimizers.Adam(params = net.parameters(),lr = 0.01)
    criterion = nn.BCELoss()    
    for i in range(epoch):
        loss = None
        net.train()
        
        for j ,(data,target) in enumerate(loaders['train']):
            data = data.float()
            target = target.float()
            optimizer.zero_grad()
            output = net(data)
            #print('flag')
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            if j % 10 == 0:
                print('epoch : {},batch : {}/60000,loss : {}'.format(
                    i+1,
                    (j+1)*128,
                    loss.item()
                ))
        history['train_loss'].append(loss)
        
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data,target in loaders['test']:
                data = data.float()
                target = target.float()
                output = net(data)
                test_loss += criterion(output,target).item()
                pred = output.argmax(dim=1,keepdim = False)
                ans = target.argmax(dim = 1,keepdim=False)
                correct += (pred == ans).sum().item()
        test_loss /= 10000
        
        print('test accuracy : {}'.format(correct/10000))
        history['test_loss'].append(test_loss)
        history['test_acc'].append(correct/10000)
    