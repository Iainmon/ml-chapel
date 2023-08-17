

record DenseLayer {
    proc backProp(grad) {
        ...
        weightsGrad += ...;
        biasGrad += ...;
        return newGrad;
    }
    proc optimize(learningRate: real) {
        weights -= learningRate * weightsGrad;
        bias -= learningRate * biasGrad;
    }
    proc resetGradients() {
        weightsGrad.data = 0.0;
        biasGrad.data = 0.0;
    }
}

record ConvLayer {
    proc backProp(grad) {
        ...
        filtersGrad += ...;
        return newGrad;
    }
    proc optimize(learningRate: real) {
        filters -= learningRate * filtersGrad;
    }
    proc resetGradients() {
        filtersGrad.data = 0.0;
    }
}

record Network {
    var layers;

    proc backward(grad: Tensor) {
        for param n in 0..#(layers.size) {
            grad = layers[layers.size - n].backProp(grad);
        }
    }
    proc optimize(learningRate: real) {
        for param layer in layers {
            layer.optimize(learningRate);
        }
    }
    proc resetGradients() {
        for param layer in layers {
            layer.resetGradients();
        }
    }
}

var net = new Network((new ConvLayer(...), new DenseLayer(...)));
const learningRate: real = 0.005;
for batch in batches {

    net.resetGradients();

    forall (input,expected) in batch (ref net) {
        var output = net.forward(input);
        var lossGradient = loss(output, expected);
        net.backward(lossGradient);
    }

    net.optimize(learningRate);
}
