import PyChai as ch
import numpy as np
import timeit
import sys

num_epochs = 100

data_size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
batch_train = (sys.argv[2].lower() in ('yes','true','t','1')) if len(sys.argv) > 2 else True
hidden_layer_size = int(sys.argv[3]) if len(sys.argv) > 3 else 4

print('data_size:', data_size)
print('batch_train:', batch_train)
print('hidden_layer_size:', hidden_layer_size)

def approx_function(x):
    center = (1,1)
    dist = np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
    if dist < 0.2:
        return np.array([1,0,0])
    elif dist < 0.4:
        return np.array([0,1,0])
    else:
        return np.array([0,0,1])



interval = np.linspace(0,2,data_size)
domain = [np.array([x,y]) for x in interval for y in interval]

data = [(x,approx_function(x)) for x in domain]

model = ch.Sequential(
    ch.Dense(16),
    ch.Sigmoid(),
    ch.Dense(16),
    ch.Sigmoid(),
    ch.Dense(16),
    ch.Sigmoid(),
    ch.Dense(3),
    ch.Sigmoid(),
)
model.forward(data[0][0])

tic = timeit.default_timer()

for e in range(num_epochs):
    np.random.shuffle(data)
    epoch_loss = 0
    model.reset_grad()
    if batch_train:
        xs = [x for x,y in data]
        ys = [y for x,y in data]
        yHats = model.forwardBatch(xs)
        deltas = [ch.loss_grad(y,yHat) for y,yHat in zip(ys,yHats)]
        model.backwardBatch(xs,deltas)
        epoch_loss = np.sum([ch.loss(y,yHat) for y,yHat in zip(ys,yHats)])
    else:
        for x,y in data:
            y_ = model.forward(x)
            delta = ch.loss_grad(y,y_)
            model.backward(x,delta)
            epoch_loss += ch.loss(y,y_)
    model.update(0.01 / len(data))
    print('epoch:', e, 'loss:', epoch_loss/len(data))

toc = timeit.default_timer()
print('time:', toc-tic)