import Chai as ch;
import Tensor as tn;
use Tensor only Tensor;
import Time;


config const dataSize = 100;
config const parallelize = true;
config const numEpochs = 100;
config const hiddenLayerSize = 4;

proc approxFunction(x: real, y: real) {
    const center = (1.0,1.0);
    const dist = sqrt((x-center[0]) ** 2.0 + (y-center[1])**2.0);
    var t = tn.zeros(3);
    if dist < 0.2 {
        t[0] = 1.0;
    } else if dist < 0.4 {
        t[1] = 1.0;
    } else {
        t[2] = 1.0;
    }
    return t;
}

proc loss(y: Tensor(1), yHat: Tensor(1)): real {
    return + reduce ((y.data - yHat.data) ** 2.0);
}

proc lossGrad(y: Tensor(1), yHat: Tensor(1)): Tensor(1) {
    return (-2.0) * (y - yHat);
}

const interval = [x in 0..#dataSize] x * (2.0 / (dataSize: real));
const sampleDomain = tn.cartesian(interval,interval);
var data: [{0..#(interval.size ** 2)}] (Tensor(1),Tensor(1));
var i = 0;
for (x,y) in sampleDomain {
    var p = tn.zeros(2);
    p[0] = x;
    p[1] = y;
    data[i] = (p,approxFunction(x, y));
    i += 1;
}

var model = new ch.Sequential(
    new ch.Dense(16),
    new ch.Sigmoid(),
    new ch.Dense(16),
    new ch.Sigmoid(),
    new ch.Dense(16),
    new ch.Sigmoid(),
    new ch.Dense(3),
    new ch.Sigmoid()
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
