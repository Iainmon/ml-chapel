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
        for x in batch:
            outputs.append(self.forward(x))
        return outputs

    def backwardBatch(self,batch,deltas):
        outputs = []
        for i in range(len(batch)):
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

    def update(self,eta):
        self.weights -= eta * self.weights_grad
        self.biases -= eta * self.biases_grad

    def reset_grad(self):
        self.weights_grad = np.zeros(self.weights.shape)
        self.biases_grad = np.zeros(self.biases.shape)





class Activation(Layer):
    def __init__(self,activation,activation_grad,vectorize=True):
        if vectorize:
            self.activation = np.vecotrize(activation)
            self.activation_grad = np.vectorize(activation_grad)
        else:
            self.activation = activation
            self.activation_grad = activation_grad

    def forward(self,x):
        return self.activation(x)
    
    def backward(self,x,delta):
        return self.activation_grad(x) * delta

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        def sigmoid_grad(x):
            return sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid,sigmoid_grad,vectorize=False)

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0,x)
        def relu_grad(x):
            return np.where(x > 0, 1, 0)
        super().__init__(relu,relu_grad,vectorize=False)

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        def tanh_grad(x):
            return 1 - np.tanh(x)**2
        super().__init__(tanh,tanh_grad,vectorize=False)

class Softmax(Activation):
    def __init__(self):
        def softmax(x):
            exp = np.exp(x)
            return exp / np.sum(exp)
        def softmax_grad(x):
            exp = np.exp(x)
            S = np.sum(exp)
            return exp * (S - exp) / (S**2)
        super().__init__(softmax,softmax_grad,vectorize=False)

class Sequential(Layer):
    def __init__(self,*layers):
        self.layers = list(layers)

    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self,x,delta):
        xs = []
        for layer in self.layers:
            xs.append(x)
            x = layer.forward(x)
        for x,layer in zip(reversed(xs),reversed(self.layers)):
            delta = layer.backward(x,delta)
        return delta

    def update(self,eta):
        for layer in self.layers:
            layer.update(eta)

    def reset_grad(self):
        for layer in self.layers:
            layer.reset_grad()

    def forwardBatch(self,batch):
        for layer in self.layers:
            batch = layer.forwardBatch(batch)
        return batch

    def backwardBatch(self,batch,deltas):
        xs = []
        for layer in self.layers:
            xs.append(batch)
            batch = layer.forwardBatch(batch)
        for x,layer in zip(reversed(xs),reversed(self.layers)):
            deltas = layer.backwardBatch(x,deltas)
        return deltas


def loss(y,y_):
    return np.sum((y-y_)**2)

def loss_grad(y,y_):
    return -2*(y-y_)

if __name__ == '__main__':

    model = Sequential(
        Dense(2),
        Dense(5),
        Dense(2)
    )

    x = np.array([1,0])
    y = np.array([0,1])
    print(model.forward(x))
    for e in range(100):
        model.reset_grad()
        y_ = model.forward(x)
        delta = loss_grad(y,y_)
        model.backward(x,delta)
        model.update(0.01)
        print('loss: ', loss(y,y_))
    print('x: ', x)
    print('y: ', y)
    print('z: ', model.forward(x))


    # x = np.array([1,0])
    # y = np.array([0,1])
    # layer = Dense(2)
    # print(layer.forward(x))
    # for e in range(100):
    #     layer.reset_grad()
    #     y_ = layer.forward(x)
    #     delta = loss_grad(y,y_)
    #     layer.backward(x,delta)
    #     layer.update(0.01)
    #     print('loss: ', loss(y,y_))
    # print('x: ', x)
    # print('y: ', y)
    # print('z: ', layer.forward(x))

    # print('-------')

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

