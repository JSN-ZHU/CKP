import os
import sys
import numpy as np
import random



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#Sigmoid activation Impl
class Sigmoid:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return sigmoid(x)

    def backward(self, dout):
        return dout * sigmoid(self.x) * (1 - sigmoid(self.x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)+1)

class Tanh:
    def __init(self):
        self.x = None
        
    def forward(self,x):
        self.x = x
        return tanh(x)
    
    def backward(self,dout):
        return 1-(tanh(dout))*(tanh(dout))
    
    
# build activation layer
class ActivationLayer:
    # param:activatetype is some activatefunction,e.g:sigmoid,relu...
    def __init__(self, activateType):
        self.activate = activateType()

    def forward(self, x):
        xx = self.activate.forward(x)
        return xx

    def backward(self, dout):
        return self.activate.backward(dout)


# build denselayer
class DenseLayer:
    def __init__(self, input_dim, hidden_nodes, lr):
        self.w = np.random.normal(0.0, 0.5, size=(input_dim, hidden_nodes))
        self.b = np.zeros(shape=(1, hidden_nodes))
        self.lr = lr
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.matmul(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.matmul(dout, self.w.T)
        nums = dout.shape[0]
        self.dw = np.matmul(self.x.T, dout) / nums
        self.db = np.mean(dout, axis=0)
        self.w = self.w - self.lr * self.dw
        self.b = self.b - self.lr * self.db
        return dx


# build sequential class
class Sequential:
    def __init__(self):
        self.layers = []
        self.layersLevel = 0
    # add layer

    def check(self):
        return len(self.layers)
        pass
    
    def remove(self,i):
        layer = self.layers[i]
        self.layers.remove(self.layers[i])
        self.layersLevel = self.layersLevel-1
        return layer
    
    def add(self, layer):
        self.layers.append(layer)
        self.layersLevel += 1
        pass

    def save(self):
        model = []
        for i in range(len(self.layers)):
            model.append(self.layers[i])
            return model,self.layersLevel
        
    # fit network
    def fit(self, x_data, y_data, epoch):
        for j in range(epoch):
            x = x_data
            for i in range(self.layersLevel):
                x = self.layers[i].forward(x)   #forward 
                

            loss1 = x - y_data
            # print(np.sum(loss))
            loss = 0.5 * (np.sum((x - y_data) ** 2))   #loss
            for i in range(self.layersLevel-1):       #backward probagation
                loss1 = self.layers[self.layersLevel - i - 1].backward(loss1)
            self.view_loss(j + 1, epoch, loss)
        pass
    # print loss every step


    def view_loss(self, step, total, loss):
        rate = step / total
        rate_num = int(rate * 40)
        r = '\rstep-%d loss value-%.4f[%s%s]\t%d%% %d/%d' % (step, loss, '>' * rate_num, '-' * (40 - rate_num),
                                                             int(rate * 100), step, total)
        sys.stdout.write(r)
        sys.stdout.flush()
        pass
    # predict by the inputdata:x_data


    def predict(self, x_data):
        x = x_data
        for i in range(self.layersLevel):
            x = self.layers[i].forward(x)
        return x
