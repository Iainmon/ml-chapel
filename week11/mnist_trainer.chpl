import Chai as ch;
import Tensor as tn;
use Tensor only Tensor;
import Time;
import MNIST;


config const dataSize = 10;
config const parallelize = true;
config const numEpochs = 100;


proc loss(y: Tensor(1), yHat: Tensor(1)): real {
    return + reduce ((y.data - yHat.data) ** 2.0);
}

proc lossGrad(y: Tensor(1), yHat: Tensor(1)): Tensor(1) {
    return (-2.0) * (y - yHat);
}
const images = MNIST.loadImages(dataSize,"./mnist/data/train-images-idx3-ubyte");
const labels = MNIST.loadLabels(dataSize,"./mnist/data/train-labels-idx1-ubyte")[1];

var data: [{0..#(dataSize)}] (Tensor(3),Tensor(1));
for i in data.domain {
    var im_ = new Tensor(images[i]);
    var im = im_.reshape(28,28,1);
    var lb = new Tensor(labels[i]);
    data[i] = (im,lb);
}

var model = new ch.Sequential(
    new ch.Conv(1,8,5),
    new ch.ReLU(),
    new ch.Conv(8,16,5),
    new ch.ReLU(),
    new ch.MaxPool(),
    new ch.Flatten(),
    new ch.Dense(80),
    new ch.ReLU(),
    new ch.SoftMax(10)
);

model.forwardProp(data[0][0]);

var tic = new Time.stopwatch();
tic.start();


for e in 0..#numEpochs {
    tn.shuffle(data);
    var epochLoss = 0.0;
    model.resetGradients();

    if parallelize {
        const xs = [d in data] d[0];
        const ys = [d in data] d[1];
        const yHats = model.forwardPropBatch(xs);
        const grads = lossGrad(ys, yHats);
        model.backwardBatch(grads, xs);
        epochLoss = + reduce loss(ys, yHats);
    } else {
        for (x,y) in data {
            var yHat = model.forwardProp(x);
            var grad = lossGrad(y, yHat);
            epochLoss += loss(y, yHat);
            model.backward(grad,x);
        }
    }

    model.optimize(0.01 / data.size);
    writeln("epoch: ", e, " loss: ", epochLoss / data.size);
}

writeln("time: ", tic.elapsed());
