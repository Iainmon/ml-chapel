import Linear as lina;
import MNIST;
import Chai as chai;
use List;
use Random;
use Math;
use IO.FormattedIO;
import Random;


writeln("Loading data...");

config const numImages = 10;
config const testSize = (numImages / 10):int;
config const testInterval = 100;
config const epochs = 8000;
config const saveInterval = 500;

config const useNewIter = false;

var images = MNIST.loadImages(numImages);
var (labels,labelVectors) = MNIST.loadLabels(numImages);
// writeln(labels);

for (i,im) in zip(images.domain,images) {
    MNIST.printImage(im);
    writeln("----------------- ", labels[i], " -----------------");
}


var imageVectorDomain = {0..#(28 * 28)};
var labelVectorDomain = {0..9};

var imageVectors: [images.domain] [imageVectorDomain] real;
for i in images.domain {
    for (m,n) in images[i].domain {
        var r = chai.rng.getNext() / 1000.0;
        imageVectors[i][m * 28 + n] = images[i][m,n]; //+ r;
    }
}


// var dataList = new list(([imageVectorDomain] real, [labelVectorDomain] real));
// for i in images.domain {
//     dataList.pushBack((imageVectors[i], labelVectors[i]));
// }
var data = [(iv,lv) in zip(imageVectors,labelVectors)] (iv,lv); // dataList.toArray();


writeln("Done loading data.");
writeln("Data: ", data.domain);


writeln("Creating network...");
// var (train,test) = chai.split(data,0.8);
const layerDimensions = [imageVectorDomain.size,200,80,labelVectorDomain.size]; // [imageVectorDomain.size, 100,100,30,labelVectorDomain.size];
var net = new chai.Network(layerDimensions);

// chai.exportModel(net,"model.bin");

// var net2 = chai.loadModel("model.bin");

writeln("Training network...");
writeln("Bias domain: ", net.biasesDomain);
writeln("Weight domain: ", net.weightsDomain);

net.save("classifier.model.bin");
writeln("Model saved.");
var net2 = chai.loadModel("classifier.model.bin");
writeln("Model loaded.");

// writeln(lina.randn(10,1));
// halt(0);

config const learningRate = 0.5; // 0.05
const decay = 0.9; // 0.1
const initialVariance = 0.1; // 0.1


const vectorizedData = [(x,y) in data] ((new lina.Vector(x)).normalize(), (new lina.Vector(y)).normalize());
const trainingData = vectorizedData[testSize..];
const testData = vectorizedData[0..#testSize];


// var cached = [(x,y) in shuffledData] (lina.vectorToMatrix(x), lina.vectorToMatrix(y));
var costDiff = 1.0;
var lastCost = 1.0;
for i in 1..epochs {
    writeln("Epoch: ", i);

//     for (x, y) in data {
//     writeln("Input: [image] Expected: ", y, " Output: ", net.feedForward(x).transpose().matrix);
// }

    // shuffle(shuffledData);
    // const cached = [(x,y) in shuffledData] (new lina.Vector(x), new lina.Vector(y));

    if i % saveInterval == 0 {
        writeln("Saving model...");
        net.save("mnist.normalized.classifier.model.bin");
        writeln("Model saved.");
    }

    if i % testInterval == 0 {
        const (correct, c, failed) = net.evaluate(testData);
        writeln("Correct: ", correct, " Failed: ", failed, " Cost: ", c);
        // for (X,Y) in testData {
        //     var Z = net.feedForward(X);
        //     writeln("Input: [image] Expected: ", Y.transpose().matrix, " Output: ", Z.transpose().matrix, " (",lina.argmax(Y.transpose())," , ", lina.argmax(Z.transpose()), ")");
        // }
    }



    if useNewIter then 
        net.updateBatchNew(trainingData,learningRate);
    else 
        net.updateBatch(trainingData,learningRate);




    var eta = initialVariance * exp(- decay * i:real);

    var cost = 0.0;
    if i % 1 == 0 {
        cost = + reduce forall (X,Y) in trainingData do net.cost(net.feedForward(X),Y);
    }
    

    var globalCost = cost / (data.domain.size: real);
    ChapelIO.writef("Cost: %20.30r\n", globalCost);

    writeln("GobalCost: ",globalCost, " (", cost, ")", " LearningRate: ", learningRate, " Eta: ", eta);

    if globalCost <= 0.005 && cost != 0.0 {
        writeln("I think that's enough...");
        break;
    }
}

writeln("Done training.");
writeln("Saving model...");
net.save("mnist.normalized.classifier.model.bin");
writeln("Model saved.");


writeln("--------------- Results ---------------");

for (x, y) in data {
    writeln("Input: [image] Expected: ", y, " Output: ", net.feedForward(x).transpose().matrix);
}

