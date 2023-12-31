import Linear as lina;
import load;
import NewChai as chai;
use List;
use Random;

use IO.FormattedIO;
import Random;


writeln("Loading data...");

const numImages = 50;

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
const layerDimensions = [imageVectorDomain.size,350,350,labelVectorDomain.size]; // [imageVectorDomain.size, 100,100,30,labelVectorDomain.size];
var net = new chai.Network(layerDimensions);


writeln("Training network...");


const learningRate = 3.0; // 0.05
const decay = 0.9; // 0.1
const initialVariance = 0.1; // 0.1
const epochs = 1000;

var shuffledData = data;
// var cached = [(x,y) in shuffledData] (lina.vectorToMatrix(x), lina.vectorToMatrix(y));
var costDiff = 1.0;
var lastCost = 1.0;
for i in 1..epochs {
    writeln("Epoch: ", i);

//     for (x, y) in data {
//     writeln("Input: [image] Expected: ", y, " Output: ", net.feedForward(x).transpose().matrix);
// }

    shuffle(shuffledData);
    var cached = [(x,y) in shuffledData] (lina.vectorToMatrix(x), lina.vectorToMatrix(y));

    for (X,Y) in cached {
        var Z = net.feedForwardM(X);
        writeln("Input: [image] Expected: ", Y.transpose().matrix, " Output: ", Z.transpose().matrix);
    }


    net.updateBatch(cached,learningRate);




    var eta = initialVariance * exp(- decay * i:real);

    var lr = learningRate;
    // if lr * eta > 0.3 {
    //     lr *= eta;
    // } else {
    //     lr = 0.3;
    // } // abs(lina.random(1,1).matrix[0,0]);//* ;

    var trainData = cached;
    // net.train(trainData, lr);
    var cost = 0.0;
    for (X,Y) in cached {
        // writeln("Input: ", x, " Expected: ", y, " Output: ", net.feedForward(x).transpose().matrix);
        // var X = lina.vectorToMatrix(x);
        // var Y = lina.vectorToMatrix(y);// .transpose().matrix;
        // var Z = lina.vectorToMatrix(z);
        // var localCost = net.costM(X,Y);
        // writeln("LocalCost: ",localCost);
        // cost += localCost;
        var Z = net.feedForwardM(X);
        cost += net.costM(Z,Y);
        // net.adjust(x, y, lr, eta);
    }

    var globalCost = cost / (data.domain.size: real);
    ChapelIO.writef("Cost: %20.30r\n", globalCost);

    writeln("GobalCost: ",globalCost, " (", cost, ")", " LearningRate: ", lr, " Eta: ", eta);

    if globalCost <= 0.005 {
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

