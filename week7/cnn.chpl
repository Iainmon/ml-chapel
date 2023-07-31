import Torch as torch;
import Tensor as tn;
use Tensor;
import Math;
import MNIST;
import Random;


var net = new torch.Network(
    (
        new torch.Conv(8),
        new torch.MaxPool(),
        new torch.SoftMax(13 * 13 * 8,10)
    )
);

// net.save("mnist.cnn.model");
// net.load("epoch_3_mnist.cnn.model");

proc forward(x: Tensor(?), lb: int) {
    const output = net.forwardProp(x);

    const loss = -Math.log(output[lb]);
    var acc = tn.argmax(output.data) == lb;

    return (output,loss,acc);
}

proc train(im: Tensor(2), lb: int, lr: real = 0.005) {
    const (output,loss,acc) = forward(im,lb);
    var gradient = tn.zeros(10);
    gradient[lb] = -1.0 / output[lb];

    net.resetGradients();
    net.backwardProp(im,gradient);
    net.optimize(lr);

    return (loss,acc);
}

config const numImages = 1000;

var imageData = MNIST.loadImages(numImages);
imageData -= 0.5;
var (trainLabels,labelVectors) = MNIST.loadLabels(numImages);

var trainImages = [im in imageData] new Tensor(im);

var trainingData = for a in zip(trainImages,trainLabels) do a;

for epoch in 0..3 {
    
    writeln("Epoch ",epoch + 1);

    var loss = 0.0;
    var numCorrect = 0;

    Random.shuffle(trainingData);

    for ((im,lb),i) in zip(trainingData,0..) {
        if i > 0 && i % 100 == 99 {
            //  print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))
            writeln("Step ",i + 1," Loss ",loss / 100," Accuracy ",numCorrect);
            loss = 0.0;
            numCorrect = 0;
        }
        const (l,a) = train(im,lb);
        loss += l;
        numCorrect += a;
    }
    // net.save("epoch_"+ epoch:string +"_mnist.cnn.model");
    
}
