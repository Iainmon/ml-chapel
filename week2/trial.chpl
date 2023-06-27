use Random;
use LinearAlgebra;
use Math;
use List;

 // Method to create a zero matrix
proc createMatrix(m: int, n: int, type eltType=real): [1..m, 1..n] eltType {
    var A: [1..m, 1..n] eltType;
    A = 0:eltType;
    return A;
}

// Method to create a random matrix
proc createRandomMatrix(m: int, n: int, type eltType=real): [1..m, 1..n] eltType {
    var A: [1..m, 1..n] eltType;
    var rng = new owned RandomStream(eltType=eltType);
    rng.fillRandom(A);
    return A;
}

class MatrixWrapper {
    var matrixDomain = {0..2, 0..2};
    var matrix: [matrixDomain] real;
    proc init(matrix: [?d] real) /*where d.rank == 2*/ {
        if d.rank != 2 {
            var m = Matrix(matrix);
            matrixDomain = m.domain;
            this.matrix = m;
        } else {
            matrixDomain = d;
            this.matrix = matrix;
        }

        // var (m,n) = matrix.domain.shape;
    }
    // proc init(matrix: [?d] real) where d.rank == 1 {

    //     // var (m,n) = matrix.domain.shape;
    // }

    proc shape {
        return this.matrix.domain.shape;
    }

    proc apply(v: [?d] real) {
        var m = transpose(Matrix(this.matrix));
        return dot(m, v);
    }

    proc mat {
        return transpose(Matrix(this.matrix));
    }

    proc vector {
        return Vector(this.matrix[0,0..]);
    }

    proc updateMatrix(m: [?d] real) {
        this.matrixDomain = d;
        this.matrix = m;
    }
}

// operator =(a: MatrixWrapper, b: [?d] real) where d.rank == 2 {
//     a.matrixDomain = d;
//     a.matrix = b;
// }


iter makeMatrices(layerSizes: [?d] int) {
    for i in d[1..] {
        var m = layerSizes[i-1];
        var n = layerSizes[i];
        var matrix = createRandomMatrix(m,n);
        yield new MatrixWrapper(matrix);
    }
}

proc access(xs: [?d] ?eltType, i: int) ref {
    return xs[d.orderToIndex((d.size + i) % d.size)];
}
proc access(ref xs: list(?etlType), i: int) ref {
    var d = xs.domain;
    return xs[d.orderToIndex((d.size + i) % d.size)];
}

class Network {
    var layerSizesDomain = {0..2};
    var layerSizes: [layerSizesDomain] int;
    var numLayers: int;

    var biasesDomain = {0..2};
    var weightsDomain = {0..1};

    var biases: [biasesDomain] (shared MatrixWrapper) = [new MatrixWrapper(createMatrix(1,1))];
    var weights: [weightsDomain] (shared MatrixWrapper) = [new MatrixWrapper(createMatrix(1,1))];

    // constructor
    proc init(layerSizes: [?d] int) {
        layerSizesDomain = d;
        this.layerSizes = layerSizes;
        this.numLayers = layerSizes.size;

        // How to initialize biases and weights?

        biasesDomain = layerSizes.domain[0..#(numLayers - 1)]; //.translate(-1);
        weightsDomain = {0..#(numLayers - 1)};

        biases = [y in biasesDomain] new shared MatrixWrapper(Vector(layerSizes[y + 1])); // [y in layerSizes[1..]] :real;
        weights = makeMatrices(layerSizes);
    }

   

    proc feedForward(a: [?d] real) {
        var yD = d;
        var y: [yD] real = a;
        for (b, w) in zip(biases, weights) {
            // var m = transpose(Matrix(w.matrix));
            var z = dot(w.mat,y);// w.apply(y); // dot(m, y);
            var ac = activation(z + b.vector);
            yD = ac.domain;
            y = ac;
        }
        return y;
    }

    proc activation(x: real): real {
        return tanh(x);
    }

    proc activationDerivative(x: real): real {
        return 1 / (cosh(x)**2);// 1 - tanh(x)**2;
    }

    proc cost(input: [?d] real, expected_output: [d] real): real {
        var output = feedForward(input);
        return 0.5 * norm(output - expected_output)**2;
    }

    proc cost_derivative(output: [?d1] real, expected_output: [?d2] real) {
        return output - expected_output;
    }

    proc backprop(x: [?d1] real, y: [?d2] real) {
        var nabla_b = [b in biases] new shared MatrixWrapper(Matrix(b.vector.size,1));
        var nabla_w = [w in weights] new shared MatrixWrapper(Matrix(w.matrix.shape[0],w.matrix.shape[1]));
        // var nabla_b = [(b.size(1), b.size(2)) in biases.domain] createMatrix(b.size(1), b.size(2));
        // var nabla_w = [(w.size(1), w.size(2)) in weights.domain] createMatrix(w.size(1), w.size(2));

        // feedforward
        var activationDomain = d1;
        var activation: [activationDomain] real = x;

        var zs: list(owned MatrixWrapper) = new list(owned MatrixWrapper);
        var activations: list(owned MatrixWrapper) = new list(owned MatrixWrapper);
        activations.append(new MatrixWrapper(activation));

        for (b, w) in zip(biases, weights) {
            var a_ = dot(w.mat, activation);
            var bv = b.vector;
            // writeln("HelloA: ", a_, " ", a_.shape, a_.domain);
            // writeln("HelloB: ", bv, " ", bv.shape, bv.domain);
            var z = dot(w.mat, activation) + b.vector;
            var a = this.activation(z);
            activationDomain = a.domain;
            activation = a;
            zs.append(new MatrixWrapper(z));
            activations.append(new MatrixWrapper(activation));
        }
        // writeln("------------------");
        var a = activations.last().vector;
        var b = zs.last().vector;
        // writeln("HelloA: ", a, " ", a.shape, a.domain);
        // writeln("HelloB: ", b, " ", b.shape, b.domain);
        // writeln("HelloY: ", y, " ", y.shape, y.domain);
        var ad = activationDerivative(zs.last().vector);
        var cd = cost_derivative(activations.last().vector, y);
        // writeln("HelloAD: ", ad, " ", ad.shape, ad.domain);
        


        var d = cost_derivative(activations.last().matrix, y) * activationDerivative(zs.last().matrix);
        var delta_ = Matrix(d);// transpose(Matrix(d));

        var deltaDomain = delta_.domain;
        var delta: [deltaDomain] real = delta_;
        
        nabla_b.last.matrix = transpose(delta); // transpose(Matrix(delta));
        nabla_w.last.matrix = dot(transpose(delta), transpose(activations[activations.size-2].matrix));


        var ac = activations[activations.size-1].matrix;
        var na = dot(delta, ac);

        // writeln("HelloAC: ", ac.shape, " ", ac.domain, ac);
        // writeln("HelloDelta: ", delta.shape, " ", delta.domain, delta);
        // writeln("HelloNA: ", na.shape, " ", na.domain, na);

        for l in 2..<numLayers {
            var z = zs[zs.size-l].vector;
            var sp = activationDerivative(z);
            var w = weights[(weights.size -l) + 1].matrix;
            writeln("HelloWeights: ", w.shape, " ", w.domain, " ", w);
            writeln("HelloDelta: ", delta.shape, " ", delta.domain, " ", delta);
            var delta_ = dot(transpose(w), delta) * Matrix(sp);
            // var delta_ = dot(transpose(weights[weights.size-l+2].matrix), delta) * activationDerivative(z);
            deltaDomain = delta_.domain;
            delta = delta_;
            writeln("HelloDelta': ", delta.shape, " ", delta.domain, " ", delta);
            var nb = access(nabla_b,-l).matrix;
            writeln("HelloNabla: ", nb.shape, " ", nb.domain, " ", nb);
            var b = access(biases,-l).vector;
            writeln("HelloBias: ", b.shape, " ", b.domain, " ", b);
            // writeln("NablaB: ", nabla_b.domain);
            access(nabla_b,-l).matrix = delta;
            access(nabla_w,-l).matrix = (dot(delta, transpose(activations[activations.size -l].matrix)));

            // access(nabla_w,-l).updateMatrix(dot(delta, activations[activations.size-l-1].matrix));
            // nabla_b[nabla_b.size-l].updateMatrix(delta);
            // nabla_w[nabla_w.size-l+1].updateMatrix(dot(delta, activations[activations.size-l-1].matrix));
        }
        return (nabla_b, nabla_w);
    }

    proc adjust(x: [?d1] real, y: [?d2] real, learningRate: real) {
        var (nabla_b, nabla_w) = backprop(x, y);
        // for (w,nw) in zip(weights,nabla_w) {
        //     w.updateMatrix(w.matrix - learningRate*nw.matrix);
        // }

        for wi in weightsDomain {
            var w = weights[wi].matrix;
            var nw = transpose(nabla_w[wi].matrix);
            // writeln("HelloWeights: ", w.shape, " ", w.domain, " ", w);
            // writeln("HelloNablaW: ", nw.shape, " ", nw.domain, " ", nw);

            var m = w - (learningRate * nw);
            this.weights[wi].matrix = m;
        }
        // writeln("HelloNablaB: ", nabla_b.domain);
        // writeln("HelloBias: ", biasesDomain);
        for bi in biasesDomain {
            var b = biases[bi].matrix;
            var nb = transpose(nabla_b[bi].matrix);
            // writeln("HelloBias: ", b.shape, " ", b.domain, " ", b);
            // writeln("HelloNablaB: ", nb.shape, " ", nb.domain, " ", nb);
            var m = b - (learningRate * nb);
            this.biases[bi].matrix = m;
        }
        // for (b,nb) in zip(biases,nabla_b) {
        //     b = b - learningRate*nb;
        // }
        // weights.matrix = [(w, nw) in zip(weights, nabla_w)] (-learningRate*nw + w.matrix);
        // biases = [(b, nb) in zip(biases, nabla_b)] (-learningRate*nb + b);



        // weights = [(-learningRate*nw + w) for (w, nw) in zip(weights, nabla_w)];
        // biases = [(-learningRate*nb + b) for (b, nb) in zip(biases, nabla_b)];
    }
    
}



var data = [
    ([0.0,0.0],[1.0,0.0,0.0,0.0]),
    ([0.0,1.0],[0.0,1.0,0.0,0.0]),
    ([1.0,0.0],[0.0,0.0,1.0,0.0]),
    ([1.0,1.0],[0.0,0.0,0.0,1.0])
];

var net = new Network([2,6,4]);
writeln(net);
writeln("Biases: ", net.biases);
writeln("Weights: ", net.weights);
for i in 1..10 {
    shuffle(data);
    for (x, y) in data {
        writeln("Input: ", x, " Expected: ", y, " Output: ", net.feedForward(x));
        net.adjust(x, y, 0.3);
    }
    writeln("Epoch: ", i);
}

for (x, y) in data {
    writeln("Input: ", x, " Expected: ", y, " Output: ", net.feedForward(x));
}



// for m in makeMatrices(layerSizes) {
//     // do something with m...
// }
