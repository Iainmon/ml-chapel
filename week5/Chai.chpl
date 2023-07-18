module Chai {

use Random;
import LinearAlgebra as LA;
use Math;
use List;
import Linear as lina;
use ChaiHelpers;
// import ChapelIO;
use IO.FormattedIO;

import IO;
import BinaryIO;
import Json;



iter makeMatrices(layerSizes: [?d] int) {
    for i in d[1..] {
        var m = layerSizes[i-1];
        var n = layerSizes[i];
        yield lina.random(n,m);
    }
}

proc access(xs: [?d] ?eltType, i: int) ref { return xs[d.orderToIndex((d.size + i) % d.size)]; }
proc access(ref xs: list(?etlType), i: int) ref {
    var d = xs.domain;
    return xs[d.orderToIndex((d.size + i) % d.size)];
}

var rng = new Random.RandomStream(eltType=real(64));


// This is a class that represents a multi-layer perceptron
class Network {
    var layerSizesDomain = {0..2};
    var layerSizes: [layerSizesDomain] int;
    var numLayers: int;

    var biasesDomain = {0..2};
    var weightsDomain = {0..1};

    // Array of bias vectors for each layer connection
    var biases: [biasesDomain] lina.Vector(real(64));

    // Array of weight matrices, one for each layer connection
    var weights: [weightsDomain] lina.Matrix(real(64));

    // Constructs a new network with the given layer sizes. Note that the first and last size must be the input and output sizes, respectively.
    proc init(layerSizes: [?d] int) {
        layerSizesDomain = d;
        this.layerSizes = layerSizes;
        this.numLayers = layerSizes.size;

        biasesDomain = layerSizes.domain[0..#(numLayers - 1)]; //.translate(-1);
        weightsDomain = {0..#(numLayers - 1)};

        // Initializes the biases and weights to be random over a normal distribution
        biases = [y in biasesDomain] lina.randn(layerSizes[y + 1],1).vectorize();
        weights = [i in weightsDomain] lina.randn(layerSizes[i + 1], layerSizes[i]); // makeMatrices(layerSizes);

    }

    // Runs the input vector through the network at it's current state and returns the output vector
    proc feedForward(in A: lina.Vector(real)): lina.Vector(real) {
        for (B, W) in zip(biases,weights) {

            // Efficient feed forward implementation
            const X = (W * A) + B;
            const v = sigmoid(X.underlyingVector); // sigmoid(Z.vector);
            A.vectorDomain = v.domain;
            A.underlyingVector = v;

            // A = v; // Try this!!
            // A = new lina.Vector(v);
        }
        return A;
    }

    iter forwardPropagationIter(in A: lina.Vector(real)) {
        for (B, W,i) in zip(biases, weights,0..) {
            const Z = (W * A) + B;
            // Efficient feed forward implementation
            const v = sigmoid(Z.underlyingVector);
            A.vectorDomain = v.domain;
            A.underlyingVector = v;
            yield (Z,A,i);
        }
    }

    proc forwardPropagation(const ref X: lina.Vector(real)) {
        var As: [0..#(biases.size + 1)] lina.Vector(real(64)); // Store all activation vectors
        var Zs: [0..#biases.size] lina.Vector(real(64));
        As[0] = X;
        for (Z,X,i) in forwardPropagationIter(X) {
            As[i + 1] = X;
            Zs[i] = Z;
        }
        return (Zs,As);
    }

    iter backpropIter(const ref X: lina.Vector(real), const ref Y: lina.Vector(real)) {
        const (Zs,As) = forwardPropagation(X);
        const AsSize = As.size;
        const ZsSize = Zs.size;

        var delta = costDerivative(As[AsSize - 1], Y) * sigmoidPrime(Zs[ZsSize - 1]);

        for l in 1..<numLayers {
            const Z = Zs[ZsSize - l];
            const A = As[AsSize - (l + 1)];
            const SP = sigmoidPrime(Z);
            const W = weights[getIdx(weights,(-l) + 1)];

            if l == 1 then
                delta = costDerivative(As[AsSize - 1], Y) * SP;
            else 
                delta = (W.transpose() * delta) * SP;
            
            const nablaB = delta;
            const nablaW = delta * A.transpose();
            yield (nablaB,nablaW);
        }
    }

    proc updateBatchNew(const ref batch: [?d] (lina.Vector(real(64)),lina.Vector(real(64))), eta: real(64)) {
        const scale = eta / batch.size;
        forall (x,y) in batch {
            for ((nablaB,nablaW),i) in zip(backpropIter(x,y),0..) {
                biases[biases.size - i - 1] -= scale * nablaB;
                weights[weights.size - i - 1] -= scale * nablaW;
            }
        }
    }

    proc backprop(const ref X: lina.Vector(real(64)), const ref Y: lina.Vector(real(64))) {

        // Arrays of the same size as the biases and weights that will store the gradients for each
        var nablaB = [b in biases] new lina.Vector(b.shape[0]);
        var nablaW = [w in weights] lina.zeros(w.shape[0],w.shape[1]);
        
        const nbSize = nablaB.size;
        const nwSize = nablaW.size;

        // Perform feed forward and preserve the intermediate actiavations and activation inputs.

        var A = X; // Current activation vector
        var As: [0..#(biases.size + 1)] lina.Vector(real(64)); // Store all activation vectors
        As[0] = A;
        var Zs: [0..#biases.size] lina.Vector(real(64)); // Store all activation inputs
        for (B, W,i) in zip(biases, weights,0..) {
            const Z = (W * A) + B;
            Zs[i] = Z;

            // Efficient feed forward implementation
            const v = sigmoid(Z.underlyingVector);
            A.vectorDomain = v.domain;
            A.underlyingVector = v;

            As[i + 1] = A;
        }

        const AsSize = As.size;
        const ZsSize = Zs.size;

        var delta = costDerivative(As[AsSize - 1], Y) * sigmoidPrime(Zs[ZsSize - 1]);

        nablaB[nbSize - 1] = delta;
        nablaW[nwSize - 1] = delta * As[AsSize - 2].transpose();
        for l in 2..<numLayers {
            const Z = Zs[ZsSize - l];
            const SP = sigmoidPrime(Z);
            const W = weights[getIdx(weights,(-l) + 1)];
            delta = (W.transpose() * delta) * SP;
            nablaB[nbSize - l] = delta;
            nablaW[nwSize - l] = delta * (As[AsSize - (l + 1)].transpose());
        }
        return (nablaB,nablaW);
    }



    proc updateBatch(const ref batch: [?d] (lina.Vector(real(64)),lina.Vector(real(64))), eta: real(64)) {
        var nablaB = [b in biases] lina.zeros(b.shape[0],1).vectorize();
        var nablaW = [w in weights] lina.zeros(w.shape[0],w.shape[1]);
        forall (x,y) in batch {
            const (deltaNablaB, deltaNablaW) = backprop(x,y);
            forall (nb,i) in zip(deltaNablaB,nablaB.domain) do nablaB[i] += nb;
            forall (nw,i) in zip(deltaNablaW,nablaW.domain) do nablaW[i] += nw;
            // nablaB += deltaNablaB; // Prefered implementation
            // nablaW += deltaNablaW;
        }
        forall (nb,i) in zip(nablaB,biases.domain) do biases[i] -= ((eta / batch.size) * nb);
        forall (nw,i) in zip(nablaW,weights.domain) do weights[i] -= ((eta / batch.size) * nw);
        // weights -= ((eta / batch.size) * nablaW); // Prefered implementation
        // biases -= ((eta / batch.size) * nablaB);
    }

    proc evaluate(const ref testData: [?d] 2*lina.Vector(real)) {
        const resultPairs = [(X,Y) in testData] (feedForward(X),Y);

        const costs = [(X,Y) in resultPairs] cost(X,Y);
        const globalCost = (+ reduce costs) / costs.size;

        const labelPairs = [(X,Y) in resultPairs] (lina.argmax(X), lina.argmax(Y));
        const results = [(x,y) in labelPairs] x == y;
        const numCorrect = + reduce results;

        const failedPairs = for (x,y) in labelPairs do if x != y then (x,y);

        return (numCorrect, globalCost, failedPairs);
    }


    // Return the cost of the network for a given input. The cost is calculated using the cross-entropy function.
    proc cost(output: lina.Vector(real(64)), expected: lina.Vector(real(64))): real(64) do
        return 0.5 * (output - expected).frobeniusNormPowTwo(); // 0.5 * ((output - expected).frobeniusNorm() ** 2.0);
    
    // Returns the gradient of the cost with respect to the output activation
    proc costDerivative(output: lina.Vector(real), expected: lina.Vector(real)) do
        return output - expected;


    // Convinience methods (unneeded)

    proc feedForward(a: [?d] real(64)): lina.Vector(real(64)) do
        return feedForward(new lina.Vector(a));

    proc costDerivative(output: [?d1] real, expected_output: [?d2] real) do
        return output - expected_output;
    
    proc cost(output: [?d] real, expected: [d] real): real(64) do
        return cost(new lina.Vector(output), new lina.Vector(expected));

}



proc main() {
    /*
    var data = [
        ([0.0,0.0],[1.0,0.0,0.0,0.0]),
        ([0.0,1.0],[0.0,1.0,0.0,0.0]),
        ([1.0,0.0],[0.0,0.0,1.0,0.0]),
        ([1.0,1.0],[0.0,0.0,0.0,1.0])
    ];

    var net = new Network([2,20,4]);
*/

    var data: [0..7] ([0..2] real(64), [0..7] real(64)) = [
        ([0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]),
        ([0.0,0.0,1.0],[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]),
        ([0.0,1.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]),
        ([0.0,1.0,1.0],[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]),
        ([1.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]),
        ([1.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]),
        ([1.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]),
        ([1.0,1.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0])
    ];
    // var d1Array = [0,0,0]:real;
    // writeln(d1Array);

    // halt(0);

    var net = new Network([3,20,8]);

    writeln(net);
    writeln("Biases: ", [ b in net.biases] b.shape);
    writeln("Weights: ", [w in net.weights] w.shape);

    for i in 0..80000 {
        var ir = (i: real(64)) / 1000.0;
        var y: real(64) = sigmoid(ir);
        ChapelIO.writef("(%dr,%20.30r)\n", ir, y);
        // writeln("(", i, ",", y ,")");
        if y == 1.0 {
            writeln("Stopped at ", i);
            break;
        }
    }

    var learningRate = 3.0;
    var patience = 10.0;
    var frustration = 0.0;
    var lastCost = 0.0;

    var shuffledData = data;
    for i in 1..10000 {
        writeln("Epoch: ", i);

        shuffle(shuffledData);
        var vectorizedData = [(x,y) in shuffledData] (lina.vectorToMatrix(x), lina.vectorToMatrix(y));
        // net.train(cached, learningRate);

        net.updateBatch(vectorizedData, learningRate);

        var cost: real(64) = 0.0;
        for (X,Y) in vectorizedData {
            writeln("Input: ", X.transpose().matrix, " Expected: ", Y.transpose().matrix, " Output: ", net.feedForward(X).transpose().matrix);
            var Z = net.feedForward(X);
            cost += net.cost(Z,Y);
        }


        var globalCost: real(64) = cost * (1.0 / (data.domain.size : real(64)));
        // writeln("Cost: ",globalCost);
        ChapelIO.writef("Cost: %20.30r\n", globalCost);

        if globalCost <= 0.005 {
            writeln("I think that's enough...");
            break;
        }
        // writeln("Frustration: ", frustration);
        // frustration += 1 / (((cost - lastCost) + 0.001) * patience);
        // lastCost = cost;
    }

    writeln("--------------- Results ---------------");

    for (x, y) in data {
        writeln("Input: ", x, " Expected: ", y, " Output: ", net.feedForward(x).transpose().matrix);
    }

}

}