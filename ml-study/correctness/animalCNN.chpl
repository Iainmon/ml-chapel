import Chai as torch;
import Tensor as tn;
use Tensor;
import Animals10;

import Math;
import AutoMath;
import Random;
import IO;
import BinaryIO;

import Time;

const numLabels = Animals10.categories.size;

// var net = new torch.Network(
//     (
//         new torch.Conv(3,20,kernelSize=3,stride=2),
//         // new torch.MaxPool(),
//         new torch.Conv(20,30,kernelSize=3,stride=2),
//         new torch.Conv(30,40,kernelSize=3),
//         new torch.MaxPool(),
//         new torch.Conv(20,25,kernelSize=3),
//         new torch.MaxPool(),
//         new torch.Flatten(),
//         new torch.Dense(80), new torch.ReLU(0.01),
//         // new torch.Dense(80), new torch.ReLU(),
//         new torch.SoftMax(numLabels)
//     )
// );
var net = new torch.Network(
    (
        new torch.Conv(3,8,4,stride=2),
        new torch.Conv(8,12,5),
        // new torch.ReLU(),
        new torch.MaxPool(),
        new torch.SoftMax(numLabels)
    )
);
// var net = new torch.Network(
//     (
//         new torch.Conv(3,16,kernelSize=3),
//         new torch.Conv(16,16,kernelSize=3),
//         new torch.MaxPool(),
//         new torch.Conv(16,32,kernelSize=3),
//         // new torch.Conv(32,32,kernelSize=5),
//         new torch.MaxPool(),
//         new torch.Conv(32,64,kernelSize=3),
//         // new torch.Conv(64,64,kernelSize=3),
//         new torch.Conv(64,128,kernelSize=3),
//         // new torch.Conv(128,128,kernelSize=3),
//         new torch.MaxPool(),
//         new torch.Conv(128,256,kernelSize=3),
//         new torch.Conv(256,256,kernelSize=3),
//         // new torch.Conv(256,256,kernelSize=3),
//         // new torch.Conv(512,512,kernelSize=3),
//         new torch.MaxPool(),
//         // new torch.Conv(256,256,kernelSize=3),
//         // new torch.Conv(256,256,kernelSize=3),
//         // new torch.MaxPool(),
//         new torch.SoftMax(10)
//     )
// );

// net.save("mnist.cnn.model");
// net.load("models/cnn/epoch_0_mnist.cnn.model");

proc forward(x: Tensor(?), lb: int) {
    const output = net.forwardProp(x);
    const loss = -Math.log(output[lb]);
    var acc = tn.argmax(output.data) == lb;

    return (output,loss,acc);
}


proc train(batch: [] (Tensor(3),int), lr: real = 0.005) {

    var loss = 0.0;
    var acc = 0;

    net.resetGradients();

    var networkGradient = net.initialGradient();

    forall ((im,lb),i) in zip(batch,0..) with (ref net,+ reduce loss, + reduce acc, + reduce networkGradient) {
        // write initializers for each layers gradientType. 
        const (output,l,a) = forward(im,lb);
        var gradient = tn.zeros(numLabels);

        const g = -1.0 / output[lb];
        if AutoMath.isnan(g) || AutoMath.isinf(g) {
            writeln("Gradient is ",g);
            writeln("Output is ",output);
            writeln("Label is ",lb);
            halt(1);
        }
        gradient[lb] = -1.0 / output[lb];
        
        net.backwardProp(im,gradient,networkGradient);

        loss += l;
        acc += if a then 1 else 0;
    }


    const batchSize = batch.domain.size;
    net.optimize(lr / batchSize, networkGradient);

    return (loss,acc);
}



config const numImages = 400;
config const batchSize = 50;
config const epochs = 80;
config const learnRate = 0.000005;

var trainingData = for (name,im) in Animals10.loadAllIter(numImages) do (im,Animals10.labelIdx(name));
forall (im,lb) in trainingData {
    im.data /= 255.0; // 225.0 ?
    im.data -= 0.5;
}
writeln(trainingData.first[0].shape);

// var trainingData = imageData;
// var imageData = Animals10.loadAll(numImages);

// imageData -= 0.5;
// var (trainLabels,labelVectors) = MNIST.loadLabels(numImages);

// var trainImages = [im in imageData] (new Tensor(im)).reshape(28,28,1);

// var trainingData = for a in zip(trainImages,trainLabels) do a;

for epoch in 0..epochs {
    
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
    writeln(forward(trainingData.first[0],trainingData.first[1]));
    //writeln(train(trainingData.first[0],trainingData.first[1],learnRate));
    //writeln(forward(trainingData.first[0],trainingData.first[1]),trainingData.first[1]);
    //writeln("Hopefully not NAN");
    // writeln(trainingData.first[0],trainingData.first[1]);
    // halt(0);

    var st = new Time.stopwatch();

    var totalTime = 0.0;
    for i in 0..#(trainingData.size / batchSize) {

        const batchRange = (i * batchSize)..#batchSize;
        const batch = trainingData[batchRange];
        
        st.clear();
        st.restart();
        
        const (loss,acc) = train(batch,learnRate);
        
        const t = st.elapsed();
        totalTime += t;

        writeln("[",i + 1," of ", trainingData.size / batchSize, "] (loss: ", loss / batchSize,") (accuracy: ", acc ," / ", batchSize, ") (time: ", t, "s)", " (avg: ", totalTime / (i + 1), "s)");

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

    net.save("models/cnn/epoch_"+ epoch:string +"_animal.cnn.model");

    
}


// proc debugFilters() {
//     var file = IO.open("week8/filters/filter.bin", IO.ioMode.cw);
//     var serializer = new BinaryIO.BinarySerializer(IO.ioendian.little);
//     var fw = file.writer(serializer=serializer);
//     // for i in 0..#8 {
//         // const fltr = net.layers[0].filters[i,..,..,0];
//         // const filter = new Tensor(fltr);
//         // filter.write(fw);
//         // writeln("Filter: ",filter);
//     // }
//     net.layers[0].filters.write(fw);
//     net.layers[1].filters.write(fw);
//     // net.layers[2].filters.write(fw);

//     fw.close();
//     file.close();

// }