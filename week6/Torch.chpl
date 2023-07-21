module Torch {
// interface Layer {}

import Linear as la;
import Math;

proc foo() ref: real {
    return 1.0;
}

// class Parameter {
//     // ref value: real;
//     // ref grad: real;

//     var value_ = foo;
//     var grad_ = foo;

//     proc value ref { return value_(); }
//     proc grad ref { return grad_(); }

//     proc init(ref v: real, ref g: real) {
//         // this.value = v;
//         // this.grad = g;
//         proc getVal() ref: real { return v; }
//         proc getGrad() ref: real { return g; }
//         value_ = getVal;
//         grad_ = getGrad;
//     }
//     proc init() {
//         // this.value = 0.0;
//         // this.grad = 0.0;
//     }
// }

// iter parameterize(ref v: la.Vector(real), ref dv: la.Vector(real)): shared Parameter {
//     for i in v.underlyingVector.domain do
//         yield new shared Parameter(v[i],dv[i]);
// }
// iter parameterize(ref m: la.Matrix(real), ref dm: la.Matrix(real)): shared Parameter {
//     for (m,n) in m.underlyingMatrix.domain do
//         yield new shared Parameter(m[m,n],dm[m,n]);
// }



class Layer {
    var layerShape: (int,int) = (0,0); // Eventually this will have to be a shape, when conisidering convolutions (tenors)
    
    var lastInput: la.Vector(real);
    var lastOutput: la.Vector(real);

    proc inputSize do return layerShape[0];
    proc outputSize do return layerShape[1];

    proc reinitialize(inputSize: int) { layerShape = (inputSize, layerShape[1]); }

    proc forward(input: la.Vector(real)): la.Vector(real) { return new la.Vector(real); };

    proc forwardMemo(const ref input: la.Vector(real)): la.Vector(real) {
        this.lastInput = input;
        this.lastOutput = this.forward(input);
        return this.lastOutput;
    }

    proc backward(delta: la.Vector(real)): la.Vector(real) { return new la.Vector(real); }

    iter parameters() ref: real {  }
    iter parametersGrad() ref: real {  }

}

// Sigmoid and Dense can be two seperate classes

class Dense: Layer {
    var bias: la.Vector(real);
    var weights: la.Matrix(real);

    var biasGrad: la.Vector(real);
    var weightsGrad: la.Matrix(real);

    proc init(outputSize: int) {
        super.init();
        layerShape = (1, outputSize);
    }
    proc init(inputSize: int, outputSize: int) {
        super.init();
        layerShape = (inputSize, outputSize);
    }

    override proc reinitialize(inputSize: int) {
        super.reinitialize(inputSize);

        bias = la.randn(outputSize,1).vectorize();
        weights = la.randn(outputSize, inputSize);
        biasGrad = la.randn(outputSize,1).vectorize();
        weightsGrad = la.randn(outputSize, inputSize);
    }

    // override iter parameters(): shared Parameter {
    //     for p in parameterize(bias, biasGrad) do yield p;
    //     for p in parameterize(weights, weightsGrad) do yield p;
    // }

    override iter parameters() ref: real {
        for i in bias.underlyingVector.domain do 
            yield bias.underlyingVector[i];
        for (m,n) in weights.underlyingMatrix.domain do
            yield weights.underlyingMatrix[m,n];
    }
    override iter parametersGrad() ref: real {
        for i in biasGrad.underlyingVector.domain do 
            yield biasGrad.underlyingVector[i];
        for (m,n) in weightsGrad.underlyingMatrix.domain do
            yield weightsGrad.underlyingMatrix[m,n];
    }

    override proc forward(input: la.Vector(real)): la.Vector(real) {
        return (weights * input) + bias;
    }

    override proc backward(delta: la.Vector(real)): la.Vector(real) {
        const newDelta = weights.transpose() * delta;
        biasGrad    = newDelta;
        weightsGrad = newDelta * lastInput.transpose();
        return newDelta;
    }

}



proc sigmoid(x: real): real do
    return 1.0 / (1.0 + Math.exp(-x));

proc sigmoidPrime(x: real): real {
    const sig = sigmoid(x);
    return sig * (1.001 - sig);
}

class Sigmoid: Layer {


    proc init() do
        super.init();

    override proc reinitialize(inputSize: int) do
        this.layerShape = (inputSize, inputSize);

    override proc forward(input: la.Vector(real)) do
        return input.fmap(sigmoid);
    
    override proc backward(delta: la.Vector(real)) {
        const SP = lastInput.fmap(sigmoidPrime);
        return delta * SP;
    }

}


class Network {
    var layersDomain: domain(1,int);
    var layers: [layersDomain] shared Layer;

    proc init(layers ...?n) {
        this.layersDomain = {0..#n};
        this.layers = layers;
        this.layers[0].reinitialize(this.layers[0].outputSize);
        for i in 1..<n {
            this.layers[i].reinitialize(this.layers[i-1].outputSize);
        }
    }

    proc feedForward(input: la.Vector(real)): la.Vector(real) {
        var output = input;
        for layer in this.layers do
            output = layer.forwardMemo(output);
        return output;
    }

    proc backpropagate(in delta: la.Vector(real)) {
        for layer in this.layers[..0] do
            delta = layer.backward(delta);
    }

    // iter parameters() {
    //     for layer in this.layers do
    //         for p in layer.parameters() do
    //             yield p;
    // }
}


class SGDOptimizer {
    var network: Network;
    var learningRate: real;
    
    proc init(network: Network, learningRate: real) {
        this.network = network;
        this.learningRate = learningRate;
    }

    proc step() {
        for layer in this.network.layers do
            for (p,dp) in zip(layer.parameters(), layer.parametersGrad()) do
                p -= this.learningRate * dp;
    }
}



proc main() {
    writeln("Hello, world!");
    var net = new shared Network(
                    new shared Dense(10,10),
                    new shared Sigmoid(),
                    new shared Dense(10,10)
                );
    // writeln(net);
    var opt = new SGDOptimizer(net, 0.01);
    net.feedForward(la.randn(10,1).vectorize());
    net.backpropagate(la.randn(10,1).vectorize());
    opt.step();
    // var parameters = net.parameters();
    // for p in parameters do
    //     writeln(p.value);


}


}
