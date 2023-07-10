# import mnist_loader
import network
import emnist
import numpy as np


# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)





images, labels = emnist.extract_test_samples('digits')
print(images.shape)
print(labels[:4])

# turn images into vectors

image_vectors = [im.reshape(28*28, 1) for im in images]

# turn labels into vectors
label_vetctors = [np.zeros((10, 1)) for l in labels]
for i, l in enumerate(label_vetctors):
    l[labels[i]] = 1.0

training_data = list(zip(image_vectors, label_vetctors))

training_data = training_data[:20000]


net = network.Network([784,200,80, 10])


# train the network

epochs = 30000
learning_rate = 0.5

for epoch in range(epochs):
    print(f'Epoch {epoch}')

    cost = 0
    for x, y in training_data:
        # calculate cost for printing

        output_activations = net.feedforward(x)
        # np.argmax(output_activations)
        # print(list(output_activations.transpose()[0]))
        local_cost = net.cost(output_activations, y)
        cost += local_cost

    net.update_mini_batch(training_data, learning_rate)

    print(f'Cost: {cost/ len(training_data)}')


# test the network

# for (x,y) in training_data:
#     output_activations = net.feedforward(x)
#     print(f'Prediction: {np.argmax(output_activations)}')
#     print(f'Actual: {np.argmax(y)}')
#     print('----')


