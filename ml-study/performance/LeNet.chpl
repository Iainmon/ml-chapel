import Chai as torch;
import Tensor as tn;
use Tensor;
import Math;
import MNIST;
import Random;
import IO;
import BinaryIO;

import MNISTTester;

var net = new torch.Network(
    (
        new torch.Conv(1,8,7),
        new torch.Conv(8,12,5),
        new torch.MaxPool(),
        new torch.SoftMax(10)
    )
);

config const numTrainImages = 2000;
config const numTestImages = 1000;

config const learnRate = 0.03; // 0.05;
config const batchSize = 100;
config const numEpochs = 60;


MNISTTester.train(
    network=net,
    numTrainImages=numTrainImages,
    numTestImages=numTestImages,
    learnRate=learnRate,
    batchSize=batchSize,
    numEpochs=numEpochs,
    savePath="../lib/models/lenet" + net.signature() + ".model"
);