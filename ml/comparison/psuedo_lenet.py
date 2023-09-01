import torch.nn as nn
import torch
import math

from torchvision import datasets, transforms

import torch.utils.data as data_utils


epochs = 50
learning_rate = 0.08
batch_size = 100
training_size = 10000
testing_size = 1000


# lenet = nn.Sequential(
#     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2,bias=False),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),

#     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0,bias=False),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),

#     nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0,bias=False),
#     nn.ReLU(),
#     nn.Flatten(),

#     nn.LazyLinear(84),
#     nn.ReLU(),
#     nn.Softmax(dim=1)
# )

# lenet = nn.Sequential(
#     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Flatten(),
#     nn.LazyLinear(120),
#     nn.ReLU(),
#     nn.LazyLinear(84),
#     nn.ReLU(),
#     nn.Softmax(dim=1)
# )

lenet = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0,bias=False),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0, bias=False),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.LazyLinear(120),
    nn.LeakyReLU(),
    nn.Softmax(dim=1)
)



train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
train_dataset = data_utils.Subset(train_dataset, torch.arange(training_size))

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
test_dataset = data_utils.Subset(test_dataset, torch.arange(testing_size))



loss_criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

def test(prnt=False):
    lenet.eval()
    correct = 0
    for data,target in test_loader:
        output = lenet(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    if prnt: print('accuracy: ', correct ,'/', len(test_dataset))
    return correct,correct / len(test_dataset)

lenet.train()


for epoch in range(epochs):
    lenet.train()
    epoch_loss = 0
    num_pairs = len(train_dataset)# mnist_images.shape[0]
    num_batches = num_pairs // batch_size
    for batch, labels in train_loader:
        # for b in range(num_batches):
        # batch = mnist_images[b*batch_size:(b+1)*batch_size,:,:]
        # batch = torch.from_numpy(batch.reshape(batch_size,1,28,28)).float() / 255
        # labels = torch.from_numpy(mnist_labels[b*batch_size:(b+1)*batch_size])

        optimizer.zero_grad()
        output = lenet(batch)
        loss = loss_criterion(output,labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(labels.data.view_as(pred)).cpu().sum().item()
        acc = correct / batch_size
        print('\r[train] epoch ', '{:3.0f}'.format(epoch + 1),' loss: ', '{:0.10f}'.format(loss.item()),' accuracy: ', '{:5.0f}'.format(correct) ,'/', batch_size, ' (', '{:5.0f}'.format(math.floor(100 * acc)), ')', end='\r')

    acc, perc = test()
    print('\n[eval]  epoch ', '{:3.0f}'.format(epoch + 1),' loss: ', '{:0.10f}'.format(epoch_loss / num_batches),' accuracy: ', '{:5.0f}'.format(acc) ,'/', len(test_dataset), ' (', '{:5.0f}'.format(math.floor(100 * perc)), ')')

