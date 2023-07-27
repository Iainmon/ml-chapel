import Tensor as tn;
use Tensor;

import Torch as torch;

import MNIST;

config const numImages = 10000;

var images = MNIST.loadImages(numImages);
var (labels,labelVectors) = MNIST.loadLabels(numImages);

var imageVectorDomain = {0..#(28 * 28)};

var imageVectors: [images.domain] [imageVectorDomain] real;
for i in images.domain {
    for (m,n) in images[i].domain {
        imageVectors[i][m * 28 + n] = images[i][m,n]: real(64);
    }
}

var trainingData = [(im,lb) in zip(imageVectors,labelVectors)] (new Tensor(im),new Tensor(lb));

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

    const epochs = 100;
    const learningRate = 0.5;

    for i in 0..epochs {
        writeln("Epoch ",i);
        const cost = net.train(trainingData,learningRate);
        writeln("Cost: ",cost);
    }

}