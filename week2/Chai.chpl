module Chai {

use Random;
import LinearAlgebra as LA;
use Math;
use List;
import Linear as lina;

// import Linear.Matrix;


proc list.get(i: int) ref {
    return this[mod(i,this.size)];
}
proc list.getIdx(i: int) ref {
    return mod(i,this.size);
}

proc getIdx(xs: [?d] ?t, i: int) {
    return mod(i,xs.domain.size);
}

iter makeMatrices(layerSizes: [?d] int) {
    for i in d[1..] {
        var m = layerSizes[i-1];
        var n = layerSizes[i];
        yield lina.random(n,m);
    }
}

proc access(xs: [?d] ?eltType, i: int) ref {
    return xs[d.orderToIndex((d.size + i) % d.size)];
}
proc access(ref xs: list(?etlType), i: int) ref {
    var d = xs.domain;
    return xs[d.orderToIndex((d.size + i) % d.size)];
}

var rng = new Random.RandomStream(eltType=real(64));

class Network {
    var layerSizesDomain = {0..2};
    var layerSizes: [layerSizesDomain] int;
    var numLayers: int;

    var biasesDomain = {0..2};
    var weightsDomain = {0..1};

    var biases: [biasesDomain] lina.Matrix(real);
    var weights: [weightsDomain] lina.Matrix(real);

    // constructor
    proc init(layerSizes: [?d] int) {
        layerSizesDomain = d;
        this.layerSizes = layerSizes;
        this.numLayers = layerSizes.size;

        // How to initialize biases and weights?

        biasesDomain = layerSizes.domain[0..#(numLayers - 1)]; //.translate(-1);
        weightsDomain = {0..#(numLayers - 1)};

        biases = [y in biasesDomain] lina.zeros(layerSizes[y + 1],1);// lina.random(layerSizes[y + 1],1);
        weights = makeMatrices(layerSizes);

        // writeln("Biases: ", biases);
        // writeln("Weights: ", weights);
    }

    proc feedForward(a: [?d] real) {
        return feedForwardM(lina.vectorToMatrix(a));
    }

    proc feedForwardM(A_: lina.Matrix(real)): lina.Matrix(real) {
        var A: lina.Matrix(real) = A_;
        for (B, W) in zip(biases,weights) {
            var X = W.dot(A) + B;
            var v = activation(X.matrix);
            A = new lina.Matrix(v); // Very slow
        }
        return A;
    }


    proc activation(x: real): real {
        return tanh(x);
    }

    proc activationM(X: lina.Matrix(real)): lina.Matrix(real) {
        var ys = activation(X.matrix);
        return new lina.Matrix(ys);
    }

    proc activationDerivative(x: real): real(64) {
        var y = tanh(x);
        return 1.0 - (y * y); // 1.0 / (cosh(x)**2.0);// 1 - tanh(x)**2;
    }

    proc activationDerivativeM(X: lina.Matrix(real)): lina.Matrix(real(64)) {
        var ys = activationDerivative(X.matrix);
        return new lina.Matrix(ys);
    }

    proc cost(input: [?d] real, expected_output: [d] real): real(64) {
        var output = feedForward(input);
        return 0.5 * norm(output - expected_output)**2.0;
    }

    proc costM(input: lina.Matrix(real(64)), expected: lina.Matrix(real(64))): real(64) {
        var output = feedForwardM(input);
        var A = output.matrix - expected.matrix;
        // return + reduce A**2.0;
        return 0.5 * LA.norm(A)**2.0;
    }

    proc costDerivative(output: [?d1] real, expected_output: [?d2] real) {
        return output - expected_output;
    }

    proc costDerivativeM(output: lina.Matrix(real), expected: lina.Matrix(real)) {
        return output - expected;
    }

    proc backprop(x: [?d1] real, y: [?d2] real) {

        var nablaB_ = [b in biases] lina.zeros(b.shape[0],b.shape[1]);
        var nablaW_ = [w in weights] lina.zeros(w.shape[0],w.shape[1]);
        
        var nablaB = new list(nablaB_);
        var nablaW = new list(nablaW_);


        var X = lina.vectorToMatrix(x);
        var Y = lina.vectorToMatrix(y);

        var A = X;
        var As: list(lina.Matrix(real)) = new list(lina.Matrix(real));
        As.append(A);
        var Zs: list(lina.Matrix(real)) = new list(lina.Matrix(real));

        for (B, W) in zip(biases, weights) {
            var Z = W.dot(A) + B;
            Zs.append(Z);
            var v = activation(Z.matrix);
            A = new lina.Matrix(v);
            As.append(A);
        }

        var delta = costDerivativeM(As.get(-1), Y) * activationDerivativeM(Zs.get(-1));
        nablaB.get(-1) = delta;
        nablaW.get(-1) = delta.dot(As.get(-2).transpose());
        
        for l in 2..<numLayers {
            var Z = Zs.get(-l);
            var SP = activationDerivativeM(Z);
            var W = weights[getIdx(weights,-l + 1)];
            delta = W.transpose().dot(delta) * SP;
            nablaB.get(-l) = delta;
            nablaW.get(-l) = delta.dot(As.get(-l - 1).transpose());
        }
    
    return (nablaB.toArray(),nablaW.toArray());

    }

    proc adjust(x: [?d1] real, y: [?d2] real, learningRate: real, scholasticScale: real = 0.01) {
        var (nablaB,nablaW) = backprop(x, y);
        for nb in nablaB {
            var nbRNG = lina.random(nb.shape[0],nb.shape[1]);
            nb += scholasticScale * nbRNG;
        }
        for nw in nablaW {
            var nwRNG = lina.random(nw.shape[0],nw.shape[1]);
            nw += scholasticScale * nwRNG;
        }
        weights = [(w,nw) in zip(weights,nablaW)] w - (learningRate * nw);
        biases = [(b,nb) in zip(biases,nablaB)] b - (learningRate * nb);
    }

    proc train(data: [?d] 2*lina.Matrix(real),eta: real) {
        var batchSize = data.size;
        var (nablaB,nablaW) = backprop(data[0][0].vector, data[0][1].vector);
        // nablaB = 0.0;
        // nablaW = 0.0;
        for (X,Y) in data {
            var x = X.vector;
            var y = Y.vector;
            var (deltaNablaB,deltaNablaW) = backprop(x,y);
            nablaB = [(nb,dnb) in zip(nablaB,deltaNablaB)] nb + dnb;
            nablaW = [(nw,dnw) in zip(nablaW,deltaNablaW)] nw + dnw;
            weights = [(w,nw) in zip(weights,nablaW)] w - ((eta / batchSize) * nw);
            biases = [(b,nb) in zip(biases,nablaB)] b - ((eta / batchSize) * nb);
        }

        // weights = [(w,nw) in zip(weights,nablaW)] w - (learningRate * nw);
        // biases = [(b,nb) in zip(biases,nablaB)] b - (learningRate * nb);
    }

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

    var data: [0..7] ([0..2] real, [0..7] real) = [
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

    var net = new Network([3,20,20,8]);

    writeln(net);
    writeln("Biases: ", net.biases);
    writeln("Weights: ", net.weights);

    var learningRate = 0.1;
    var patience = 10.0;
    var frustration = 0.0;
    var lastCost = 0.0;

    var shuffledData = data;
    for i in 1..10000 {
        writeln("Epoch: ", i);

        shuffle(shuffledData);
        var cached = [(x,y) in shuffledData] (lina.vectorToMatrix(x), lina.vectorToMatrix(y));
        // net.train(cached, learningRate);

        var cost = 0.0;
        for (x, y) in shuffledData {
            // writeln("Input: ", x, " Expected: ", y, " Output: ", net.feedForward(x).transpose().matrix);
            var X = lina.vectorToMatrix(x);
            var Y = lina.vectorToMatrix(y);
            cost += net.costM(X,Y);
            net.adjust(x, y, learningRate);
        }


        var globalCost = cost / data.domain.size;
        writeln("Cost: ",globalCost);

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