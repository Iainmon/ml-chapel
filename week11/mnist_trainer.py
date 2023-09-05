import PyChai as ch
import numpy as np
import timeit
import sys
import emnist


num_epochs = 100

data_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10

print('data_size:', data_size)



interval = np.linspace(0,2,data_size)
domain = [np.array([x,y]) for x in interval for y in interval]
images,labels = emnist.extract_training_samples('digits')

data = []
for i in range(data_size):
    l = labels[i]
    lb = np.zeros(10)
    lb[l] = 1
    im = np.expand_dims(images[i,:,:],axis=2)
    data.append((im,lb))
print(data[0][0].shape)


model = ch.Sequential(
    ch.Conv(1,8,5),
    ch.Tanh(),
    ch.Conv(8,16,5),
    ch.Tanh(),
    ch.MaxPool(2),
    ch.Flatten(),
    ch.Dense(80),
    ch.Tanh(),
    ch.Dense(10),
    ch.Softmax()
)

model.forward(data[0][0])

tic = timeit.default_timer()

for e in range(num_epochs):
    np.random.shuffle(data)
    epoch_loss = 0
    model.reset_grad()
    xs = [x for x,y in data]
    ys = [y for x,y in data]
    yHats = model.forwardBatch(xs)
    deltas = [ch.loss_grad(y,yHat) for y,yHat in zip(ys,yHats)]
    model.backwardBatch(xs,deltas)
    epoch_loss = np.sum([ch.loss(y,yHat) for y,yHat in zip(ys,yHats)])
    # else:
    #     for x,y in data:
    #         y_ = model.forward(x)
    #         delta = ch.loss_grad(y,y_)
    #         model.backward(x,delta)
    #         epoch_loss += ch.loss(y,y_)
    model.update(0.01 / len(data))
    print('epoch:', e, 'loss:', epoch_loss/len(data))

toc = timeit.default_timer()
print('time:', toc-tic)