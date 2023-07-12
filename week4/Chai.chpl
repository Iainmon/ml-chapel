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

class Network {
    var layerSizesDomain = {0..2};
    var layerSizes: [layerSizesDomain] int;
    var numLayers: int;

    var biasesDomain = {0..2};
    var weightsDomain = {0..1};

    var biases: [biasesDomain] lina.Matrix(real(64));
    var weights: [weightsDomain] lina.Matrix(real(64));

    // constructor
    proc init(layerSizes: [?d] int) {
        layerSizesDomain = d;
        this.layerSizes = layerSizes;
        this.numLayers = layerSizes.size;

        // How to initialize biases and weights?

        biasesDomain = layerSizes.domain[0..#(numLayers - 1)]; //.translate(-1);
        weightsDomain = {0..#(numLayers - 1)};

        biases = [y in biasesDomain] lina.randn(layerSizes[y + 1],1);
        // weights = [i in weightsDomain] (1.0 / (layerSizes[i + 1] * layerSizes[i])) * lina.random(layerSizes[i + 1], layerSizes[i]); // makeMatrices(layerSizes);

        weights = [i in weightsDomain] lina.randn(layerSizes[i + 1], layerSizes[i]); // makeMatrices(layerSizes);

        // writeln("Biases: ", biases);
        // writeln("Weights: ", weights);
    }

    proc feedForward(a: [?d] real(64)) {
        return feedForwardM(lina.vectorToMatrix(a));
    }

    proc feedForwardM(A_: lina.Matrix(real(64))): lina.Matrix(real(64)) {
        var A: lina.Matrix(real(64)) = A_;
        for (B, W) in zip(biases,weights) {
            var X = W.dot(A) + B;
            var v = sigmoid(X.matrix);
            A = new lina.Matrix(v); // Very slow
        }
        return A;
    }


    proc backprop(X: lina.Matrix(real(64)), Y: lina.Matrix(real(64))) {

        // var X = lina.vectorToMatrix(x);
        // var Y = lina.vectorToMatrix(y);


        var nablaB_ = [b in biases] lina.zeros(b.shape[0],b.shape[1]);
        var nablaW_ = [w in weights] lina.zeros(w.shape[0],w.shape[1]);
        
        var nablaB = new list(nablaB_);
        var nablaW = new list(nablaW_);


        var A = X;
        var As: list(lina.Matrix(real(64))) = new list(lina.Matrix(real(64)));
        As.pushBack(A);
        var Zs: list(lina.Matrix(real(64))) = new list(lina.Matrix(real(64)));

        for (B, W) in zip(biases, weights) {
            var Z = W.dot(A) + B;
            Zs.pushBack(Z);
            var v = sigmoid(Z.matrix);
            A = new lina.Matrix(v);
            As.pushBack(A);
        }

        var delta = costDerivativeM(As.get(-1), Y) * sigmoidPrimeM(Zs.get(-1));
        nablaB[nablaB.getIdx(-1)] = delta.copy();
        nablaW[nablaW.getIdx(-1)] = delta.dot(As.get(-2).transpose());

        // nablaB.get(-1) = delta;
        // nablaW.get(-1) = delta.dot(As.get(-2).transpose());
        
        for l in 2..<numLayers {
            var Z = Zs.get(-l);
            var SP = sigmoidPrimeM(Z);
            var W = weights[getIdx(weights,(-l) + 1)];
            delta = W.transpose().dot(delta) * SP;
            nablaB[nablaB.getIdx(-l)] = delta.copy();
            nablaW[nablaW.getIdx(-l)] = delta.dot(As.get((-l) - 1).transpose());
            // nablaB.get(-l) = delta;
            // nablaW.get(-l) = delta.dot(As.get((-l) - 1).transpose());
        }
    
        return (nablaB.toArray(),nablaW.toArray());

    }

    proc updateBatch(batch: [?d] (lina.Matrix(real(64)),lina.Matrix(real(64))), eta: real(64)) {
        var nablaB = [b in biases] lina.zeros(b.shape[0],b.shape[1]);
        var nablaW = [w in weights] lina.zeros(w.shape[0],w.shape[1]);
        for (x,y) in batch {
            var (deltaNablaB, deltaNablaW) = backprop(x,y);
            forall (nb,i) in zip(deltaNablaB,nablaB.domain) do nablaB[i] += nb;
            forall (nw,i) in zip(deltaNablaW,nablaW.domain) do nablaW[i] += nw;
            // nablaB += deltaNablaB;
            // nablaW += deltaNablaW;
            // nablaB = [(nb,dnb) in zip(nablaB,deltaNablaB)] nb + dnb;
            // nablaW = [(nw,dnw) in zip(nablaW,deltaNablaW)] nw + dnw;
        }
        forall (nb,i) in zip(nablaB,biases.domain) do biases[i] -= ((eta / batch.size) * nb);
        forall (nw,i) in zip(nablaW,weights.domain) do weights[i] -= ((eta / batch.size) * nw);
        
        // weights -= ((eta / batch.size) * nablaW);
        // biases -= ((eta / batch.size) * nablaB);
        // weights = [(w,nw) in zip(weights,nablaW)] w - ((eta / batch.size) * nw);
        // biases = [(b,nb) in zip(biases,nablaB)] b - ((eta / batch.size) * nb);
    }

    // proc adjust(x: lina.Matrix(real), y: lina.Matrix(real), learningRate: real) {
    //     var (nablaB,nablaW) = backprop(x, y);
    //     // for nb in nablaB {
    //     //     var nbRNG = lina.random(nb.shape[0],nb.shape[1]);
    //     //     nb += scholasticScale * nbRNG;
    //     // }
    //     // for nw in nablaW {
    //     //     var nwRNG = lina.random(nw.shape[0],nw.shape[1]);
    //     //     nw += scholasticScale * nwRNG;
    //     // }
    //     weights = [(w,nw) in zip(weights,nablaW)] w - (learningRate * nw);
    //     biases = [(b,nb) in zip(biases,nablaB)] b - (learningRate * nb);
    // }

    // proc train(data: [?d] 2*lina.Matrix(real),eta: real) {
    //     var batchSize = data.size;
    //     nablaB = [b in biases] lina.zeros(b.shape[0],b.shape[1]);
    //     nablaW = [w in weights] lina.zeros(w.shape[0],w.shape[1]);
    //     // var (nablaB,nablaW) = backprop(data[0][0].vector, data[0][1].vector);
    //     // nablaB = 0.0;
    //     // nablaW = 0.0;
    //     for (X,Y) in data {
    //         var x = X.vector;
    //         var y = Y.vector;
    //         var (deltaNablaB,deltaNablaW) = backprop(x,y);
    //         nablaB = [(nb,dnb) in zip(nablaB,deltaNablaB)] nb + dnb;
    //         nablaW = [(nw,dnw) in zip(nablaW,deltaNablaW)] nw + dnw;
    //         weights = [(w,nw) in zip(weights,nablaW)] w - ((eta / batchSize) * nw);
    //         biases = [(b,nb) in zip(biases,nablaB)] b - ((eta / batchSize) * nb);
    //     }

    //     // weights = [(w,nw) in zip(weights,nablaW)] w - (learningRate * nw);
    //     // biases = [(b,nb) in zip(biases,nablaB)] b - (learningRate * nb);
    // }

    proc cost(output: [?d] real, expected: [d] real): real(64) {
        const outputM = lina.vectorToMatrix(output);
        const expectedM = lina.vectorToMatrix(expected);
        return costM(outputM, expectedM);
    }

    proc costM(output: lina.Matrix(real(64)), expected: lina.Matrix(real(64))): real(64) {
        return 0.5 * (output - expected).frobeniusNormPowTwo(); // 0.5 * ((output - expected).frobeniusNorm() ** 2.0);
    }

    proc costDerivative(output: [?d1] real, expected_output: [?d2] real) {
        return output - expected_output;
    }

    proc costDerivativeM(output: lina.Matrix(real), expected: lina.Matrix(real)) {
        return output - expected;
    }
/*
    override proc serialize(writer: IO.fileWriter(?), ref serializer: writer.serializerType) throws {
        writer.write(this.layerSizesDomain);
        writer.write(this.layerSizes);
        writer.write(this.numLayers);
        writer.write(this.biasesDomain);
        writer.write(this.weightsDomain);
        writer.write(this.biases);
        writer.write(this.weights);
    }
    proc init(reader: IO.fileReader(?), ref deserializer: reader.deserializerType) throws {
        // this.layerSizesDomain = reader.read(deserializer=deserializer);
        try! {
            this.layerSizesDomain = deserializer.deserializeType(reader,domain(1,int));
            this.layerSizes = deserializer.deserializeType(reader,[this.layerSizesDomain] int);
            this.numLayers = deserializer.deserializeType(reader,int);
            this.biasesDomain = deserializer.deserializeType(reader,domain(1,int));
            this.weightsDomain = deserializer.deserializeType(reader,domain(1,int));
            this.biases = deserializer.deserializeType(reader,[this.biasesDomain] lina.Matrix(real));
            this.weights = deserializer.deserializeType(reader,[this.weightsDomain] lina.Matrix(real));
        }
    }
*/
}
/*
proc exportModel(net: Network, filename: string) {
    try {
        var serializer = new BinaryIO.BinarySerializer();
        // var serializer = new Json.JsonSerializer();

        var fw = IO.openWriter(filename, serializer=serializer);
        fw.write(net);// net.serialize(fw, serializer);
        fw.close();
        writeln("Model exported to ", filename);
    } catch e {
        writeln("Could not export model to ", filename);
        writeln("Error: ", e);
    }

}

proc loadModel(filename: string) {
    try {
        var deserializer = new BinaryIO.BinaryDeserializer();

        // var deserializer = new Json.JsonDeserializer();
        var fr = IO.openReader(filename, deserializer=deserializer);
        var net = deserializer.deserializeType(fr, Network);
        fr.close();
        return net;
    } catch e {
        writeln("Could not load model from ", filename);
        writeln("Error: ", e);
        halt(0);
    }
}

*/


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
            writeln("Input: ", X.transpose().matrix, " Expected: ", Y.transpose().matrix, " Output: ", net.feedForwardM(X).transpose().matrix);
            var Z = net.feedForwardM(X);
            cost += net.costM(Z,Y);
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