import Chai as chai;
import MNISTTester2;
import Tensor as tn;
use Tensor only Tensor;

var net = new chai.Network(
    (
        new chai.Conv(1,16,5),
        new chai.ReLU(0.1),
        new chai.MaxPool(),
        new chai.Conv(16,32,5),
        new chai.ReLU(0.1),
        new chai.MaxPool(),
        new chai.Flatten(),
        new chai.Dense(200),
        new chai.ReLU(0.1),
        new chai.SoftMax(10)
    )
);

config const numTrainImages = 10000;
config const numTestImages = 1000;

config const learnRate = 0.03; // 0.05;
config const batchSize = 32;
config const numEpochs = 20;


MNISTTester2.train(
    network=net,
    numTrainImages=numTrainImages,
    numTestImages=numTestImages,
    learnRate=learnRate,
    batchSize=batchSize,
    numEpochs=numEpochs,
    savePath="../lib/models/test_cnn" + net.signature() + ".model",
    watch=true
);
