import Chai as torch;
import Tensor as tn;
use Tensor;
import Math;
import MNIST;
import Random;
import IO;
import BinaryIO;

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

// net.save("mnist.cnn.model");
// net.load("models/cnn/epoch_0_mnist.cnn.model");

proc forward(x: Tensor(?), lb: int) {
    const output = net.forwardProp(x);

    const loss = -Math.log(output[lb]);
    var acc = tn.argmax(output.data) == lb;

    return (output,loss,acc);
}

proc train(im: Tensor(?), lb: int, lr: real = 0.005) {
    const (output,loss,acc) = forward(im,lb);
    var gradient = tn.zeros(10);
    gradient[lb] = -1.0 / output[lb];

    net.resetGradients();
    net.backwardProp(im,gradient);
    net.optimize(lr);

    return (loss,acc);
}

proc train(data: [] (Tensor(3),int), lr: real = 0.005) {
    // writeln("Training on ",data.domain.size," images");
    const size = data.domain.size;

    var loss = 0.0;
    var acc = 0;

    net.resetGradients();
    forall ((im,lb),i) in zip(data,0..) with (ref net,+ reduce loss, + reduce acc) {
        const (output,l,a) = forward(im,lb);
        var gradient = tn.zeros(10);
        gradient[lb] = -1.0 / output[lb];
        
        net.backwardProp(im,gradient);

        loss += l;
        acc += if a then 1 else 0;
    }

    net.optimize(lr);

    return (loss,acc);
}



config const numImages = 500;
config const learnRate = 0.005; // 0.05;
config const batchSize = 1;

var imageData = MNIST.loadImages(numImages);
imageData -= 0.5;
var (trainLabels,labelVectors) = MNIST.loadLabels(numImages);

var trainImages = [im in imageData] (new Tensor(im)).reshape(28,28,1);

var trainingData = for a in zip(trainImages,trainLabels) do a;

for epoch in 0..12 {
    
    writeln("Epoch ",epoch + 1);


    Random.shuffle(trainingData);

    // debugFilters();


    // var loss = 0.0;
    // var numCorrect = 0;
    // for ((im,lb),i) in zip(trainingData,0..) {
    //     if i > 0 && i % 100 == 99 {
    //         //  print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))
    //         writeln("Step ",i + 1," Loss ",loss / 100," Accuracy ",numCorrect);
    //         loss = 0.0;
    //         numCorrect = 0;
    //     }
    //     const (l,a) = train(im,lb);
    //     loss += l;
    //     numCorrect += a;
    // }

    for i in 0..#(trainingData.size / batchSize) {
        const batchRange = (i * batchSize)..#batchSize;
        const batch = trainingData[batchRange];
        const (loss,acc) = train(batch,learnRate);
        writeln("[",i + 1," of ", trainingData.size / batchSize, "] Loss ", loss / batchSize," Accuracy ", acc ," / ", batchSize);
        
        // if loss < 0.00001 {
        //     net.save("models/cnn/epoch_"+ epoch:string +"_mnist.cnn.model");
        //     halt(0);
        // }

    }

    writeln("Evaluating...");

    var loss = 0.0;
    var numCorrect = 0;

    forall (im,lb) in trainingData with (+ reduce loss, + reduce numCorrect) {
        const (o,l,a) = forward(im,lb);
        loss += l;
        numCorrect += a;
    }

    writeln("End of epoch ", epoch + 1, " Loss ", loss / trainingData.size, " Accuracy ", numCorrect, " / ", trainingData.size);

    net.save("models/cnn/epoch_"+ epoch:string +"_mnist.cnn.model");

    
}


proc debugFilters() {
    var file = IO.open("week8/filters/filter.bin", IO.ioMode.cw);
    var serializer = new BinaryIO.BinarySerializer(IO.ioendian.little);
    var fw = file.writer(serializer=serializer);
    // for i in 0..#8 {
        // const fltr = net.layers[0].filters[i,..,..,0];
        // const filter = new Tensor(fltr);
        // filter.write(fw);
        // writeln("Filter: ",filter);
    // }
    net.layers[0].filters.write(fw);
    net.layers[1].filters.write(fw);
    // net.layers[2].filters.write(fw);

    fw.close();
    file.close();

}