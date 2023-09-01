import Chai as chai;
import Tensor as tn;
use Tensor only Tensor;
import Math;
import Time;

config param perfTest = false;

tn.seedRandom(0);

config const epochs = 5000;
config const learnRate = 0.3;

var net = new chai.Network(
    (
        new chai.Dense(10),
        new chai.Sigmoid(),
        new chai.Dense(10),
        new chai.Sigmoid(),
        new chai.Dense(1),
        new chai.Sigmoid()
    )
);




proc forward(batch: [] (Tensor(1),Tensor(1))) {
    var outputs: [0..#batch.size] Tensor(1);
    var losses: [0..#batch.size] real;
    var lossesGrad: [0..#batch.size] Tensor(1);

    forall ((input,expected),idx) in zip(batch,0..) with (ref net) {
        const output = net.forwardProp(input);
        outputs[idx] = output;
        losses[idx] = (+ reduce ((expected - output).data ** 2.0)) / output.domain.size;
        lossesGrad[idx] = expected - output;
    }
    return (losses, lossesGrad, outputs);
}

proc train(batch: [] (Tensor(1),Tensor(1)), lr:real) {

    var (losses,deltas,outputs) = forward(batch);
    var inputs = [(i,o) in batch] i;


    net.resetGradients();
    net.backwardPropBatch(inputs,deltas);
    net.optimize(-lr / batch.size);

    const loss = (+ reduce losses) / batch.size;

    return loss;
}

proc test(batch: [] (Tensor(1),Tensor(1))) {
    const (losses,lossesGrad, outputs) = forward( batch);
    for ((input,expected),output) in zip(batch,outputs) {
        writeln(input.data, " -> ", output.data, " [", expected.data ,"]");
    }
}


var inputs: [0..#4] Tensor(1);
inputs[0] = [0.0,0.0];
inputs[1] = [0.0,1.0];
inputs[2] = [1.0,0.0];
inputs[3] = [1.0,1.0];


var outputs: [0..#4] Tensor(1);
outputs[0] = [0.0];
outputs[1] = [1.0];
outputs[2] = [1.0];
outputs[3] = [0.0];



var batch = for a in zip(inputs,outputs) do a;
net.forwardPropBatch(inputs);

var batchSizes = [2,3];

var t = new Time.stopwatch();
t.start();

for e in 1..epochs {
    tn.shuffle(batch);
    tn.shuffle(batchSizes);
    const loss = train(batch[0..#(batchSizes[0])],learnRate);
}
const loss = train(batch[0..#(batchSizes[0])],learnRate);

writeln("Epoch: ", e, " Loss: ", AutoMath.floor(loss * 10000) / 10000);


test(batch);

if perfTest then writeln("time: ", t.elapsed());
