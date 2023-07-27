import Tensor as tn;
use Tensor;

import Torch as torch;

import MNIST;

config const numImages = 10;
config const testSize = (numImages / 10):int;
config const testInterval = 100;
config const epochs = 8000;
config const saveInterval = 500;

var images = MNIST.loadImages(numImages);
var (labels,labelVectors) = MNIST.loadLabels(numImages);

var imageVectorDomain = {0..#(28 * 28)};

var imageVectors: [images.domain] [imageVectorDomain] real;
for i in images.domain {
    for (m,n) in images[i].domain {
        imageVectors[i][m * 28 + n] = images[i][m,n]: real(64);
    }
}

const data = [(im,lb) in zip(imageVectors,labelVectors)] (new Tensor(im),new Tensor(lb));
const trainingData = data[testSize..];
const testData = data[0..#testSize];


proc main() {
    var net = new torch.Network(
            (
                new torch.Dense(28 * 28,200),
                new torch.Sigmoid(200),
                new torch.Dense(200,70),
                new torch.Sigmoid(70),
                new torch.Dense(70,10),
                new torch.Sigmoid(10)
            )
        );

    const learningRate = 0.5;

    for i in 1..epochs {
        writeln("Epoch ",i);
        const cost = net.train(trainingData,learningRate);
        writeln("Cost: ",cost);
    }

}

writeln("Done training.");