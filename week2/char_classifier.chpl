import Linear as lina;
import load;
import Chai as chai;
use List;
use Random;

import Random;


writeln("Loading data...");

const numImages = 20;

var images = load.loadImages(numImages);
var (labels,labelVectors) = load.loadLabels(numImages);
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
const layerDimensions = [imageVectorDomain.size, 50,labelVectorDomain.size];
var net = new chai.Network(layerDimensions);


writeln("Training network...");


const learningRate = 0.05;
const decay = 0.1;
const epochs = 100000;

var shuffledData = data;
// var cached = [(x,y) in shuffledData] (lina.vectorToMatrix(x), lina.vectorToMatrix(y));

var lastCost = 1.0;
for i in 1..epochs {
    writeln("Epoch: ", i);

//     for (x, y) in data {
//     writeln("Input: [image] Expected: ", y, " Output: ", net.feedForward(x).transpose().matrix);
// }

    shuffle(shuffledData);
    var cached = [(x,y) in shuffledData] (lina.vectorToMatrix(x), lina.vectorToMatrix(y));

    var lr = learningRate; //* abs(lina.random(1,1).matrix[0,0]);//* ;
    var trainData = cached;
    // net.train(trainData, lr);

    var cost = 0.0;
    for ((x, y),(X,Y)) in zip(shuffledData, cached) {
        // writeln("Input: ", x, " Expected: ", y, " Output: ", net.feedForward(x).transpose().matrix);
        // var X = lina.vectorToMatrix(x);
        // var Y = lina.vectorToMatrix(y);// .transpose().matrix;
        // var Z = lina.vectorToMatrix(z);
        // var localCost = net.costM(X,Y);
        // writeln("LocalCost: ",localCost);
        // cost += localCost;
        cost += net.costM(X,Y);
        net.adjust(x, y, lr,0.1 * exp(- decay * i:real));
    }

    var globalCost = cost / (data.domain.size: real);
    writeln("GobalCost: ",globalCost, " (", cost, ")", " LearningRate: ", lr);

    if globalCost <= 0.005 {
        writeln("I think that's enough...");
        break;
    }

    lastCost = globalCost;
    // writeln("Frustration: ", frustration);
    // frustration += 1 / (((cost - lastCost) + 0.001) * patience);
    // lastCost = cost;
}

writeln("--------------- Results ---------------");

for (x, y) in data {
    writeln("Input: [image] Expected: ", y, " Output: ", net.feedForward(x).transpose().matrix);
}

