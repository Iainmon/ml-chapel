
import Chai as torch;
import Tensor as tn;
use Tensor;
import Math;
import MNIST;
import Random;
import IO;
import BinaryIO;


tn.seedRandom(0);

config const modelPath = "../performance/data/mnist_cnn_epoch_14.model";
config const numImages = 30000;

var net = new torch.Network(
    (
        new torch.Conv(1,12,3,stride=2),
        new torch.Conv(12,16,4),
        // new torch.ReLU(),
        new torch.MaxPool(),
        new torch.SoftMax(10)
    )
);

net.load(modelPath);

proc forward(x: Tensor(?), lb: int) {
    const output = net.forwardProp(x);
    const loss = -Math.log(output[lb]);
    const acc = tn.argmax(output.data) == lb;
    return (output,loss,acc);
}

var imageRawData = MNIST.loadImages(numImages,"../lib/mnist/data/train-images-idx3-ubyte");
imageRawData -= 0.5;
const (labels,labelVectors) = MNIST.loadLabels(numImages,"../lib/mnist/data/train-labels-idx1-ubyte");


const images = [im in imageRawData] (new Tensor(im)).reshape(28,28,1);
const testingData = for a in zip(images,labels) do a;

var loss = 0.0;
var acc = 0;

forall (im,lb) in testingData with (+ reduce loss, + reduce acc) {
    const (output,loss_,acc_) = forward(im,lb);
    loss += loss_;
    acc += if acc_ then 1 else 0;
}

loss /= numImages;

writeln("Loss: ",loss," Accuracy: ",acc ," / ", numImages, " ", (acc * 100):real / (numImages:real), " %");






