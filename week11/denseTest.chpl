import Chai as ch;
import Tensor as tn;
use Tensor only Tensor;

proc loss(y: Tensor(1), yHat: Tensor(1)): real {
    var dif = y.data - yHat.data;
    var squares = dif ** 2.0;
    return + reduce squares;
}

proc lossGrad(y: Tensor(1), yHat: Tensor(1)): Tensor(1) {
    return (-2.0) * (y - yHat);
}

var layer = new ch.Sequential(
    new ch.Dense(2),
    new ch.Sigmoid(),
    new ch.Dense(5),
    new ch.Sigmoid(),
    new ch.Dense(2)
);
var x = tn.zeros(2);
x[0] = 1.0;
var y = tn.zeros(2);
y[1] = 1.0;

layer.forwardProp(x);

for e in 0..100 {
    layer.resetGradients();
    var yHat = layer.forwardProp(x);
    var l = loss(y, yHat);
    var lGrad = lossGrad(y, yHat);
    layer.backwardBatch([lGrad],[x]);
    layer.optimize(0.01);
    writeln("Epoch: ", e, " Loss: ", l);
}
writeln("yHat: ", layer.forwardProp(x));
