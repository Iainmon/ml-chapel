import numpy as np

class Layer:
    def forward(self,x):
        pass
    def backward(self,x,delta):
        pass

    def update(self,eta):
        pass
    
    def reset_grad(self):
        pass

    def forwardBatch(self,batch):
        outputs = []
        for i in range(batch.shape[0]):
            x = batch[i]
            outputs.append(self.forward(x))
        return outputs

    def backwardBatch(self,batch,deltas):
        outputs = []
        for i in range(batch.shape[0]):
            x = batch[i]
            delta = deltas[i]
            outputs.append(self.backward(x,delta))
        return outputs
    


    
class Dense(Layer):
    def __init__(self,output_size):
        self.output_size = output_size
        self.uninitialized = True

    def initialize(self,input_size):
        self.weights = np.random.randn(self.output_size,input_size) / np.sqrt(input_size)
        self.biases = np.zeros(self.output_size)
        self.weights_grad = np.zeros(self.weights.shape)
        self.biases_grad = np.zeros(self.biases.shape)
        self.uninitialized = False
    
    def forward(self,x):
        if self.uninitialized: self.initialize(x.shape[0])
        return np.dot(self.weights,x) + self.biases

    def backward(self,x,delta):
        self.weights_grad += np.outer(delta,x)
        self.biases_grad += delta
        return np.dot(self.weights.T,delta)

        # curr_out = self.forward(x)
        # self.weights_grad += np.dot(delta,x.T)
        # self.biases_grad += delta
        # return np.dot(self.weights.T,delta)


    def update(self,eta):
        self.weights -= eta * self.weights_grad
        self.biases -= eta * self.biases_grad

    def reset_grad(self):
        self.weights_grad = np.zeros(self.weights.shape)
        self.biases_grad = np.zeros(self.biases.shape)


def loss(y,y_):
    return np.sum((y-y_)**2)

def loss_grad(y,y_):
    return -2*(y-y_)

if __name__ == '__main__':
    x = np.array([1,0])
    y = np.array([0,1])
    layer = Dense(2)
    print(layer.forward(x))
    for e in range(100):
        layer.reset_grad()
        y_ = layer.forward(x)
        delta = loss_grad(y,y_)
        layer.backward(x,delta)
        layer.update(0.01)
        print('loss: ', loss(y,y_))
    print('x: ', x)
    print('y: ', y)
    print('z: ', layer.forward(x))

    print('-------')

    # weights = np.random.randn(3,2)
    # x = np.random.randn(2)
    # y = np.dot(weights,x)
    # print(weights,weights.shape)
    # print(x,x.shape)
    # print(y,y.shape)

    # x = np.array([1,0])
    # dy = np.random.randn(2)
    # print('x: ', x)
    # print('dy: ', dy)
    # print('outer: ', np.outer(dy,x))
    # print('matmul: ', np.matmul(dy[...,None],x[...,None].T))

    # np.dot(x[...,None],y[None,...]) == np.outer(x,y)
    # np.matmul(x[...,None],y[None,...].T) == np.outer(x,y)

