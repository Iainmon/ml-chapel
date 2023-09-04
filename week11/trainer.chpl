import Chai as ch;
import Tensor as tn;
use Tensor only Tensor;

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
    var dif = y.data - yHat.data;
    var squares = dif ** 2.0;
    return + reduce squares;
}

proc lossGrad(y: Tensor(1), yHat: Tensor(1)): Tensor(1) {
    return (-2.0) * (y - yHat);
}

const interval = [x in 0..#100] x * (2.0 / 100.0);
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

var model = new ch.Network(
    new ch.Dense(4),
    new ch.Sigmoid(),
    new ch.Dense(3),
    new ch.Sigmoid()
);

model.forwardProp(data[0][0]);


for e in 0..#1000 {
    tn.shuffle(data);
    var epochLoss = 0.0;
    model.resetGradients();
    for (x,y) in data {
        var yHat = model.forwardProp(x);
        var grad = lossGrad(yHat, y);
        epochLoss += loss(y, yHat);
        model.backwardProp(x,grad);
    }
    model.optimize(0.01 / data.size);
    writeln("epoch: ", e, " loss: ", epochLoss / data.size);
}

