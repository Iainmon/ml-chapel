
import Chai as chai;
import Tensor as tn;
use Tensor;
import Math;
import MNIST;
import Random;
import IO;
import BinaryIO;
import MNISTTester;


tn.seedRandom(0);

config const modelDir = "../lib/models/";
config const numImages = 30000;

var net1 = new chai.Network(
    (
        new chai.Conv(1,12,3,stride=2),
        new chai.Conv(12,16,4),
        new chai.MaxPool(),
        new chai.SoftMax(10)
    )
);

MNISTTester.test(
    network=net1,
    numImages=numImages,
    modelPath= modelDir + "mnist" + net1.signature() + ".model"
);

var net2 = new chai.Network(
    (
        new chai.Conv(1,32,5,stride=2),
        new chai.Conv(32,64,5,stride=1),
        new chai.MaxPool(),
        new chai.SoftMax(10)
    )
);

MNISTTester.test(
    network=net2,
    numImages=numImages,
    modelPath=modelDir + "mnist" + net2.signature() + ".model"
);




