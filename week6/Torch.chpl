
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

        var numFilters: int;
        var filters: Tensor(3);
        var filtersGrad: Tensor(3);

        proc init(numFilters: int) {
            this.numFilters = numFilters;
            this.filters = tn.randn(numFilters,3,3) / 9.0;
            this.filtersGrad = tn.zeros(numFilters,3,3);
        }

        iter regions(image: Tensor(2)) {
            const (h,w) = image.shape;
            for i in 0..#(h-2) {
                for j in 0..#(w-2) {
                    var region = image[i..i+3, j..j+3];
                    yield (region,i,j);
                }
            }
        }


        proc forwardProp(image: Tensor(2)): Tensor(3) {
            const (h,w) = image.shape;

            var output = tn.zeros(h-2,w-2,numFilters);

            // for (region,i,j) in regions(x) {
            //     // var filterResults = new Tensor(numFilters,3,3);
            //     // var filterResults = [i in 0..#numFilters] region * new Tensor(filters[i,..,..]);
            //     var convSums: [0..#numFilters] real;
            //     forall k in 0..#numFilters {
            //         const filter = filters.data[k,..,..];
            //         const conv = region * filter;
            //         convSums[k] = + reduce conv;
            //     }
            //     output[i,j,..] = convSums;
            // }
            forall i in 0..#(h-2) with (ref output) {
                forall j in 0..#(w-2) with (ref output) {
                    const region = image[i..#3, j..#3];
                    forall k in 0..#numFilters with (ref output) {
                        const filter = filters.data[k,..,..];
                        const conv = region * filter;
                        output.data[i,j,k] = + reduce conv;
                    }
                }
            }
            return output;
        }

        proc backward(delta: Tensor(3), image: Tensor(2)): Tensor(2) {
            const (h,w) = image.shape;
            var output = tn.zeros(h,w);

            forall i in 0..#(h-2) with (ref output) {
                forall j in 0..#(w-2) with (ref output) {
                    const region = image[i..#3, j..#3];
                    forall k in 0..#numFilters with (ref output) {
                        const filter = filters.data[k,..,..];
                        const conv = region * filter;
                        filtersGrad.data[k,..,..] += delta.data[i,j,k] * region;
                        output.data[i,j] += delta.data[i,j,k] * + reduce conv;
                    }
                }
            }
            return output;
        }

        proc optimize(mag: real(64)) {
            filters -= mag * filtersGrad;
        }
    }

    record MaxPool {

        iter regions(convs: Tensor(3)) {
            const (h,w,numFilters) = convs.shape;
            const newH: int = h / 2;
            const newW: int = w / 2;

            for i in 0..#newH {
                for j in 0..#newW {
                    var region = convs[i*2..#2, j*2..#2, ..];
                    yield (region,i,j);
                }
            }
        }

        proc forwardProp(convs: Tensor(3)): Tensor(3) {
            const (h,w,numFilters) = convs.shape;
            const newH: int = h / 2;
            const newW: int = w / 2;

            var output = tn.zeros(newH,newW,numFilters);
            for (region,i,j) in regions(convs) {
                forall k in 0..#numFilters with (ref output) {
                    output[i,j,k] = max reduce region[..,..,k];
                }
            }
            return output;
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
                if i % ((data.size / 100):int + 1) == 0 {
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

        var n2 = new Network(
            (
                new Conv(3),
                new MaxPool()
            )
        );
        const image = tn.randn(10,10);
        writeln(image);
        const convs = n2.forwardProp(image);
        writeln(convs);
        // n2.train([(image,convs)],0.5);

    }
}