
module Torch {
    
    // import Tensor.Tensor;
    import Tensor as tn;
    import IO;

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
            // lastInput = x;
            const activation = (weights * x) + bias;
            // lastOutput = activation;
            return activation;
        }

        proc backward(delta: Tensor(1)): Tensor(1) {
            const newDelta = weights.transpose() * delta;
            biasGrad    = newDelta;
            weightsGrad = newDelta * lastInput.transpose();
            return newDelta;
        }

        proc backward(delta: Tensor(1),lastInput: Tensor(1)): Tensor(1) {
            const newDelta = weights.transpose() * delta;
            biasGrad    += newDelta;
            weightsGrad += newDelta * lastInput.transpose();
            return newDelta;
        }

        proc optimize(mag: real(64)) {
            bias -= mag * biasGrad;
            weights -= mag * weightsGrad;
        }
    }

    record Sigmoid {
        var lastInput: Tensor(1);
        var lastOutput: Tensor(1);

        // var grad: Tensor(1);

        proc init(size: int) {
            lastInput = tn.zeros(size);
            lastOutput = tn.zeros(size);
            // grad = tn.zeros(size);
        }

        proc forwardProp(x: Tensor(1)): Tensor(1) {
            // lastInput = x;
            const activation = tn.sigmoid(x);
            // lastOutput = activation;
            return activation;
        }

        proc backward(delta: Tensor(1)): Tensor(1) {
            const sp = tn.sigmoidPrime(lastInput);
            const grad = delta * sp;
            return grad; 
        }

        proc backward(delta: Tensor(1),lastInput: Tensor(1)): Tensor(1) {
            const sp = tn.sigmoidPrime(lastInput);
            const grad = delta * sp;
            return grad; 
        }

        proc optimize(mag: real) {
            
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

    proc forwardPropHelp(ref layers, param n: int, x: Tensor(?)) {
        if n == layers.size then return x;

        const xNext = layers[n].forwardProp(x);
        return forwardPropHelp(layers, n+1, xNext);
    }

    proc backwardPropHelp(ref layers, param n: int, x: Tensor(?)) {
        if n == 0 then return layers[0].backward(x);

        const xNext = layers[n].backward(x);
        return backwardPropHelp(layers, n-1, xNext);
    }

    proc backwardForwardPropHelp(ref layers, param n: int, x: Tensor(?), lastDelta: Tensor(?)) {
        // if n == layers.size then return lastDelta;

        const lastInput = layers[n].forwardProp(x);
        if n == layers.size - 1 then
            return layers[n].backward(lastDelta,x);

        const delta = backwardForwardPropHelp(layers, n+1, lastInput, lastDelta);
        return layers[n].backward(delta,x);
    }

    record Network {
        var layers;

        proc init(layers) {
            this.layers = layers;
        }

        proc forwardProp(x: Tensor(?)) {
            return forwardPropHelp(this.layers, 0, x);
        }
        proc backwardProp(x: Tensor(?)) {
            return backwardPropHelp(this.layers,this.layers.size - 1,x);
        }
        proc optimize(mag: real) {
            for param i in 0..#(layers.size) {
                layers[i].optimize(mag);
            }
        }

        proc cost(x: Tensor(?), y: Tensor(?)) {
            const z = this.forwardProp(x);
            return tn.frobeniusNormPowTwo(y - z);
        }

        proc optimize(x: Tensor(?),y: Tensor(?),mag: real) {
            const z = this.forwardProp(x);
            const delta = z - y;
            backwardForwardPropHelp(this.layers,0,x,delta);
            // this.optimize(mag);
        }

        proc train(data, learningRate: real) {
            var cost = 0.0;
            forall ((x,y),i) in zip(data,0..) with (ref this, ref cost) {
                // this.optimize(x,y,learningRate);
                const z = this.forwardProp(x);
                const delta = z - y;
                cost += tn.frobeniusNormPowTwo(delta);
                backwardForwardPropHelp(this.layers,0,x,delta);
                if i % (data.size / 100):int == 0 {
                    // try! IO.stderr.write(">");
                    write(">");
                    try! IO.stdout.flush();
                }

            }
            // try! IO.stderr.writeln();
            writeln();
            cost /= data.size;
            writeln("Optimizing...");
            for param i in 0..#(layers.size) {
                layers[i].optimize(learningRate / data.size);
            }

            return cost;

        }
    }

    proc main() {
        var n = new Network(
            (
                new Dense(3,3),
                new Sigmoid(3),
                new Dense(3,6),
                new Sigmoid(6)
            )
        );

        const inv: [0..#3] real = [1,2,3];
        const input = new Tensor(inv);

        var output = n.forwardProp(input);
        var reversedInput = n.backwardProp(output);
        
        writeln(input);
        writeln(output);
        writeln(reversedInput);

        const t = tn.randn(3,4);
        writeln(t);

        // var shape = (3,4,5);
        // for i in 0..<(3 * 4 * 5) {
        //     writeln(i, " ", tn.nbase(shape,i));
        // }

    }
}