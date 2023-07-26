
module Torch {
    
    // import Tensor.Tensor;
    import Tensor as tn;

    use Tensor;
    // import Tensor;
    

    record Dense {
        var bias: Tensor(1);
        var weights: Tensor(2);


        var biasGrad: Tensor(1);
        var weightsGrad: Tensor(2);

        var lastInput: Tensor(1);
        var lastOutput: Tensor(1);


        proc init(inputSize: int, outputSize: int) {
            bias = tn.randn(outputSize);
            weights = tn.randn(outputSize, inputSize);

            biasGrad = tn.zeros(outputSize);
            weightsGrad = tn.zeros(outputSize, inputSize);

            lastInput = tn.zeros(inputSize);
            lastOutput = tn.zeros(outputSize);
        }

        proc forwardProp(x: Tensor(1)): Tensor(1) {
            lastInput = x;
            const activation = (weights * x) + bias;
            lastOutput = activation;
            return activation;
        }
    }

    record Sigmoid {
        var lastInput: Tensor(1);
        var lastOutput: Tensor(1);

        var grad: Tensor(1);

        proc init(size: int) {
            lastInput = tn.zeros(size);
            lastOutput = tn.zeros(size);
            grad = tn.zeros(size);
        }

        proc forwardProp(x: Tensor(1)): Tensor(1) {
            lastInput = x;
            const activation = tn.sigmoid(x);
            lastOutput = activation;
            return activation;
        }
    }

    record Conv {

        proc forwardProp(x: Tensor(2)): Tensor(2) {
            return new Tensor(real, 0, 0);
        }
    }

    record MaxPool {

        proc forwardProp(x: Tensor(2)): Tensor(1) {
            return new Tensor(real, 0);
        }
    }

    proc forwardPropHelp(const ref layers, param n: int, x: Tensor(?)) {
        if n == layers.size-1 then return x;

        var xNext = layers[n].forwardProp(x);
        return forwardPropHelp(layers, n+1, xNext);
    }

    record Network {
        var layers;

        proc init(layers) {
            this.layers = layers;
        }

        proc forwardProp(x: Tensor(?)) {
            return forwardPropHelp(this.layers, 0, x);
        }
    }

    proc main() {
        var n = new Network(
            (
                new Dense(2,3),
                new Sigmoid(3),
                new Dense(3,1),
                new Sigmoid(1)
            )
        );

        var p = n.forwardProp(new Tensor(2));

        writeln(p);
    }
}