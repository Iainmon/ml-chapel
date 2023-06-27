import math
import numpy as np

class Network:

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation(np.dot(w, a) + b)
        return a

    def activation(self, x):
        return np.tanh(x)

    def activation_derivative(self, x):
        return 1 - np.tanh(x)**2

    def cost(self,input,expected_output):
        output = self.feed_forward(input)
        return 0.5*np.linalg.norm(output-expected_output)**2 # Mean Squared Error cost function

    def cost_derivative(self, output, expected_output):
        return (output - expected_output) # derivative of cost function

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * self.activation_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        print("NablaB: ", nabla_b[-1].shape, nabla_b[-1])
        print("NablaW: ", nabla_w[-1].shape, nabla_w[-1])
        exit()
        ac = activations[-2].transpose()
        na = np.dot(delta, activations[-2].transpose())
        # print("--------------------------")
        # print("AC: ", ac.shape, ac)
        # print("Delta: ", delta.shape, delta)
        # print("--------------------------")
        # print("NA: ", na.shape, na)
        # print("--------------------------")
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_derivative(z)
            w = self.weights[-l+1].transpose()
            print("Weight: ", w.shape, w)
            
            print("Delta: ", delta.shape, delta)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            print("Delta': ", delta.shape, delta)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def adjust(self, x, y, learning_rate):
        nabla_b, nabla_w = self.backprop(x, y)
        self.weights = [w-(learning_rate)*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate)*nb for b, nb in zip(self.biases, nabla_b)]

    def train(self, data, epochs, learning_rate):
        for epoch in range(epochs):
            # print(f'epoch: {epoch}')
            cost = 0
            np.random.shuffle(data)
            for i, o in data:
                # cost += self.cost(i, o)
                self.adjust(i,o,learning_rate)
            # print(f'cost: {cost / len(data)}')


nbits = 4

def int_to_bin(n):
    return np.array([int(x) for x in format(n, f'0{math.floor(math.log2(nbits))}b')]).reshape(-1, 1)

def bin_to_int(bin_arr):
    return int(''.join(str(x[0]) for x in bin_arr), 2)

def int_to_bin_class(n):
    return np.array([int(1 if i == n else 0) for i in range(nbits)]).reshape(-1,1)

# Generate the training data
# data = [(int_to_bin(n), int_to_bin(n)) for n in range(8)]
data = [(int_to_bin(n), int_to_bin_class(n)) for n in range(nbits)]



# Instantiate the neural network
nn = Network([math.floor(math.log2(nbits)),6, nbits])
print("biases:")
for i in range(len(nn.biases)):
    print(nn.biases[i].shape, nn.biases[i])
print("--------")
print("weights:")
for i in range(len(nn.weights)):
    print(nn.weights[i].shape, nn.weights[i])

# # Test the network
# for n in range(nbits):
#     x = int_to_bin(n)
#     y = nn.feed_forward(x)
#     print(f'{n}: {y}')

# Training parameters
epochs = 10000
learning_rate = 0.5


nn.train(data,epochs,learning_rate)

# Training loop
for epoch in range(epochs):
    np.random.shuffle(data)
    for x, y in data:
        nn.adjust(x, y, learning_rate)


# Test the network
for n in range(nbits):
    x = int_to_bin(n)
    y = nn.feed_forward(x)
    print(f'{n}: {y}')
    # print(f'{n}: {bin_to_int(y.round().astype(int))}  |  {y}')


# for i,o in data:
#     print(i,o)

# print(nn.biases[0].shape)
# print(nn.weights[0].shape)