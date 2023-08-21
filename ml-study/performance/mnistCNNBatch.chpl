import Chai as torch;
import Tensor as tn;
use Tensor;
import Math;
import MNIST;
import Random;
import IO;
import BinaryIO;


tn.seedRandom(0);

config const dataPath = "ml-study/performance/data";

// var net = new torch.Network(
//     (
//         new torch.Conv(1,20,3),
//         new torch.MaxPool(),
//         // new torch.SoftMax(13 * 13 * 8,10)
//         new torch.Conv(20,10,3),
//         new torch.MaxPool(),
//         new torch.SoftMax(5 * 5 * 10,10)
//     )
// );

// var net = new torch.Network(
//     (
//         new torch.Conv(1,6,5),   // 24
//         new torch.MaxPool(),     // 12
//         new torch.Conv(6,16,5),  // 8
//         new torch.MaxPool(),     // 4
//         new torch.SoftMax(4 * 4 * 16,10)
//     )
// );


// This works
// var net = new torch.Network(
//     (
//         new torch.Conv(1,8,3),
//         new torch.MaxPool(),
//         new torch.SoftMax(13 * 13 * 8,10)
//     )
// );

// var net = new torch.Network(
//     (
//         new torch.Conv(1,8,3),
//         new torch.MaxPool(),
//         new torch.Conv(8,8,3),
//         new torch.MaxPool(),
//         new torch.SoftMax(5 * 5 * 8,10)
//     )
// );

// // THIS IS MY BENCHMARK
// var net = new torch.Network(
//     (
//         new torch.Conv(1,8,7),
//         new torch.Conv(8,12,5),
//         new torch.MaxPool(),
//         new torch.SoftMax(10)
//     )
// );

var net = new torch.Network(
    (
        new torch.Conv(1,8,4,stride=2),
        new torch.Conv(8,12,5),
        // new torch.ReLU(),
        new torch.MaxPool(),
        new torch.SoftMax(10)
    )
);

proc forward(x: Tensor(?), lb: int) {
    const output = net.forwardProp(x);
    const loss = -Math.log(output[lb]);
    const acc = tn.argmax(output.data) == lb;
    return (output,loss,acc);
}

proc train(data: [] (Tensor(3),int), lr: real = 0.005) {
    // writeln("Training on ",data.domain.size," images");
    const size = data.domain.size;

    var loss = 0.0;
    var acc = 0;

    net.resetGradients();
    var gradients: [0..#size] Tensor(1,real);

    forall ((im,lb),i) in zip(data,0..) with (ref net,+ reduce loss, + reduce acc) {
        const (output,l,a) = forward(im,lb);
        var gradient = tn.zeros(10);
        gradient[lb] = -1.0 / output[lb];
        
        gradients[i] = gradient;
        // net.backwardProp(im,gradient);

        loss += l;
        acc += if a then 1 else 0;
    }
    const inputs = [im in data] im[0];
    net.backwardPropBatch(inputs,gradients);

    net.optimize(lr / size);

    return (loss,acc);
}



config const numTrainImages = 500;
config const numTestImages = 100;

config const learnRate = 0.5; // 0.05;
config const batchSize = 1;
config const numEpochs = 30;


const numImages = numTrainImages + numTestImages;

var imageRawData = MNIST.loadImages(numImages);
imageRawData -= 0.5;
var (labels,labelVectors) = MNIST.loadLabels(numImages);


var images = [im in imageRawData] (new Tensor(im)).reshape(28,28,1);
var labeledImages = for a in zip(images,labels) do a;

tn.shuffle(labeledImages);

var trainingData = labeledImages[0..#numTrainImages];
var testingData = labeledImages[numTrainImages..#numTestImages];


for epoch in 0..#numEpochs {
    
    writeln("Epoch ",epoch + 1);
    net.forwardProp(trainingData[0][0]);

    tn.shuffle(trainingData);

    for i in 0..#(trainingData.size / batchSize) {
        const batchRange = (i * batchSize)..#batchSize;
        const batch = trainingData[batchRange];
        const (loss,acc) = train(batch,learnRate);
        writeln("[",i + 1," of ", trainingData.size / batchSize, "] Loss ", loss / batchSize," Accuracy ", acc ," / ", batchSize);

    }

    writeln("Evaluating...");

    var loss = 0.0;
    var numCorrect = 0;

    forall (im,lb) in testingData with (+ reduce loss, + reduce numCorrect) {
        const (o,l,a) = forward(im,lb);
        loss += l;
        numCorrect += a;
    }

    writeln("End of epoch ", epoch + 1, " Loss ", loss / testingData.size, " Accuracy ", numCorrect, " / ", testingData.size);

    net.save( dataPath + "/mnist_cnn_epoch_" + epoch:string + ".model");
}