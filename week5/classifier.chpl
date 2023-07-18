import Linear as lina;
import MNIST;
import Chai as chai;
use List;
use Random;
use Math;
use IO.FormattedIO;
import Random;


writeln("Loading data...");

config const numImages = 50;
config const testSize = 10;
config const testInterval = 10;

var images = MNIST.loadImages(numImages);
var (labels,labelVectors) = MNIST.loadLabels(numImages);
writeln(labels);


var imageVectorDomain = {0..#(28 * 28)};
var labelVectorDomain = {0..9};

var imageVectors: [images.domain] [imageVectorDomain] real;
for i in images.domain {
    for (m,n) in images[i].domain {
        var r = chai.rng.getNext() / 1000.0;
        imageVectors[i][m * 28 + n] = images[i][m,n] + r;
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

// writeln(lina.randn(10,1));
// halt(0);

config const learningRate = 0.8; // 0.05
const decay = 0.9; // 0.1
const initialVariance = 0.1; // 0.1
const epochs = 8000;


const trainingData = [(x,y) in data[testSize..]] (new lina.Vector(x), new lina.Vector(y));
const testData = [(x,y) in data[0..#testSize]] (new lina.Vector(x), new lina.Vector(y));

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

    if i % testInterval == 0 {
        const (correct, c, failed) = net.evaluate(testData);
        writeln("Correct: ", correct, " Failed: ", failed, " Cost: ", c);
        // for (X,Y) in testData {
        //     var Z = net.feedForward(X);
        //     writeln("Input: [image] Expected: ", Y.transpose().matrix, " Output: ", Z.transpose().matrix, " (",lina.argmax(Y.transpose())," , ", lina.argmax(Z.transpose()), ")");
        // }
    }



    net.updateBatch(trainingData,learningRate);




    var eta = initialVariance * exp(- decay * i:real);

    var lr = learningRate;
    // if lr * eta > 0.3 {
    //     lr *= eta;
    // } else {
    //     lr = 0.3;
    // } // abs(lina.random(1,1).matrix[0,0]);//* ;

    // net.train(trainData, lr);
    var cost = 0.0;
    if i % 1 == 0 {
        for (X,Y) in trainingData {
            // writeln("Input: ", x, " Expected: ", y, " Output: ", net.feedForward(x).transpose().matrix);
            // var X = lina.vectorToMatrix(x);
            // var Y = lina.vectorToMatrix(y);// .transpose().matrix;
            // var Z = lina.vectorToMatrix(z);
            // var localCost = net.costM(X,Y);
            // writeln("LocalCost: ",localCost);
            // cost += localCost;
            var Z = net.feedForward(X);
            cost += net.cost(Z,Y);
            // net.adjust(x, y, lr, eta);
        }
    }
    

    var globalCost = cost / (data.domain.size: real);
    ChapelIO.writef("Cost: %20.30r\n", globalCost);

    writeln("GobalCost: ",globalCost, " (", cost, ")", " LearningRate: ", lr, " Eta: ", eta);

    if globalCost <= 0.005 && cost != 0.0 {
        writeln("I think that's enough...");
        break;
    }
    costDiff = abs(globalCost - lastCost);
    lastCost = globalCost;
    // writeln("Frustration: ", frustration);
    // frustration += 1 / (((cost - lastCost) + 0.001) * patience);
    // lastCost = cost;
}

writeln("--------------- Results ---------------");

for (x, y) in data {
    writeln("Input: [image] Expected: ", y, " Output: ", net.feedForward(x).transpose().matrix);
}

