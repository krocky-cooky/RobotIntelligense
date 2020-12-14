import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optimizers
from sklearn.metrics import accuracy_score

x_train = np.load('./datasets/kmnist-train-imgs.npz')['arr_0']
t_train = np.load('./datasets/kmnist-train-labels.npz')['arr_0']
x_test = np.load('./datasets/kmnist-test-imgs.npz')['arr_0']
t_test = np.load('./datasets/kmnist-test-labels.npz')['arr_0']

t_train = np.identity(10)[t_train]
t_test = np.identity(10)[t_test]
x_train = x_train.reshape((60000,-1))
x_test = x_test.reshape((10000,-1))
x_train = x_train/255
x_test = x_test/255

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
    ):
        super().__init__()
        self.l1 = nn.Linear(input_dim,hidden_dim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim,output_dim)
        self.a2 = nn.Sigmoid()
        
        
        self.layers = [self.l1,self.a1,self.l2,self.a2]
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x



if __name__ == '__main__':
    torch.manual_seed(123)
    device = torch.device('cpu')
    model = MLP(784,50,10).to(device)

    criterion = nn.BCELoss()
    optimizer = optimizers.SGD(model.parameters(),lr=0.1)

    def compute_loss(t,y):
        return criterion(y,t)

    def train_step(x,t):
        model.train()
        preds = model(x)
        #print(preds.shape)
        loss = compute_loss(t,preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def test_step(x,t):
        x = torch.Tensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.eval()
        preds = model(x)
        loss = compute_loss(t,preds)
        return loss,preds

    epochs = 10000
    batch_size = 500
    n_batches = x_train.shape[0]//batch_size

    for epoch in range(epochs):
        train_loss = 0.
        x_ = x_train.copy()
        t_ = t_train.copy()
        x_ = torch.Tensor(x_).to(device)
        t_ = torch.Tensor(t_).to(device)
        
        for n_batch in range(n_batches):
            start = n_batch*batch_size
            end = start + batch_size
            loss = train_step(x_[start:end],t_[start:end])
            train_loss += loss.item()
            
        print('epoch : {}, loss : {:.3}'.format(
            epoch + 1,
            train_loss
        ))


        

    loss,preds = test_step(x_test,t_test)
    test_loss = loss.item()
    preds = np.argmax(preds.data.cpu().numpy(),axis=1)
    ans = np.argmax(t_test,axis=1)
    test_acc = accuracy_score(ans,preds)

    print('test_loss : {:.3f}, test_acc : {:.3f}'.format(
        test_loss,
        test_acc
    ))