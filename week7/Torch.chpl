
module Torch {
    
    // import Tensor.Tensor;
    import Tensor as tn;
    import IO;
    import BinaryIO;

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
        proc resetGradients() {
            biasGrad.data = 0;
            weightsGrad.data = 0;
        }

        proc write(fw: IO.fileWriter) throws {
            bias.write(fw);
            weights.write(fw);
        }
        proc read(fr: IO.fileReader) throws {
            bias.read(fr);
            weights.read(fr);
        }
    }

    record Sigmoid {
        proc init(size: int) { }

        proc forwardProp(x: Tensor(1)): Tensor(1) {
            const activation = tn.sigmoid(x);
            return activation;
        }

        proc backward(delta: Tensor(1),lastInput: Tensor(1)): Tensor(1) {
            const sp = tn.sigmoidPrime(lastInput);
            const grad = delta * sp;
            return grad; 
        }

        proc optimize(mag: real) { }
        proc resetGradients() { }

        proc write(fw: IO.fileWriter) throws {
            // fw.write("[sigmoid]");
        }
        proc read(fr: IO.fileReader) throws { }

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

            // Python analog (slow)
            /* var output = tn.zeros(h-2,w-2,numFilters);
            for (region,i,j) in regions(x) {
                // var filterResults = new Tensor(numFilters,3,3);
                // var filterResults = [i in 0..#numFilters] region * new Tensor(filters[i,..,..]);
                var convSums: [0..#numFilters] real;
                forall k in 0..#numFilters {
                    const filter = filters.data[k,..,..];
                    const conv = region * filter;
                    convSums[k] = + reduce conv;
                }
                output[i,j,..] = convSums;
            }*/

            // Using tensor type (faster?)
            /*var output = tn.zeros(h-2,w-2,numFilters);
            forall i in 0..#(h-2) with (ref output) {
                forall j in 0..#(w-2) with (ref output) {
                    const region = image[i..#3, j..#3];
                    forall k in 0..#numFilters with (ref output) {
                        const filter = filters.data[k,..,..];
                        const conv = region * filter;
                        output.data[i,j,k] = + reduce conv;
                    }
                }
            }*/

            // Using arrays (optimal)
            var convs: [0..#(h-2), 0..#(w-2), 0..#numFilters] real;
            forall (i,j,k) in convs.domain {
                const region = image[i..#3, j..#3];
                const filter = filters.data[k,..,..];
                const conv = region * filter;
                convs[i,j,k] = + reduce conv;
            }
            const output = new Tensor(convs);

            return output;
        }

        proc backward(delta: Tensor(3), image: Tensor(2)): Tensor(2) {
            const (h,w) = image.shape;

            // Using tensor type (faster?)
            /* var output = tn.zeros(h,w);
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
            }*/
            
            // Using arrays (optimal)
            var grad: [0..#h, 0..#w] real;
            forall (i,j,k) in delta.data.domain with (ref this) {
                const region = image[i..#3, j..#3];
                const filter = filters[k,..,..];
                const conv = region * filter;
                filtersGrad[k,..,..] += delta[i,j,k] * region;
                grad[i,j] += delta[i,j,k] * + reduce conv;
            }
            const output = new Tensor(grad);

            return output;
        }

        proc optimize(mag: real(64)) {
            filters -= mag * filtersGrad;
        }
        proc resetGradients() {
            filtersGrad.data = 0;
        }

        proc write(fw: IO.fileWriter) throws {
            fw.write(numFilters);
            filters.write(fw);
        }

        proc read(fr: IO.fileReader) throws {
            var nf = fr.read(int);
            if nf != numFilters then tn.err("Conv read: numFilters mismatch");
            filters.read(fr);
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

            // Python analog (slow)
            /*var output = tn.zeros(newH,newW,numFilters);
            for (region,i,j) in regions(convs) {
                forall k in 0..#numFilters with (ref output) {
                    output[i,j,k] = max reduce region[..,..,k];
                }
            }*/

            // Using tensor type (faster?)
            /*var output = tn.zeros(newH,newW,numFilters);
            forall i in 0..#newH with (ref output) {
                forall j in 0..#newW with (ref output) {
                    const region = convs[i*2..#2, j*2..#2, ..];
                    forall k in 0..#numFilters with (ref output) {
                        output[i,j,k] = max reduce region[..,..,k];
                    }
                }
            }*/

            // Using arrays (optimal)
            var pools: [0..#newH, 0..#newW, 0..#numFilters] real;
            forall (i,j,k) in pools.domain {
                const region = convs[i*2..#2, j*2..#2, ..];
                pools[i,j,k] = max reduce region[..,..,k];
            }
            const output = new Tensor(pools);

            return output;
        }

        proc argmax(m: [?d] real) where d.rank == 2 {
            var max: real = m[d.first];
            var maxIndex: 2*int = d.first;
            for (i,j) in m.domain {
                if m[i,j] > max {
                    max = m[i,j];
                    maxIndex = (i,j);
                }
            }
            return maxIndex - d.first;
        }

        proc backward(delta: Tensor(3), convs: Tensor(3)): Tensor(3) {
            const (h,w,numFilters) = convs.shape;

            const newH: int = h / 2;
            const newW: int = w / 2;

            var grad: [0..#h, 0..#w, 0..#numFilters] real;
            forall (i,j,k) in delta.data.domain {
                const region = convs[i*2..#2, j*2..#2, k];
                const (maxI,maxJ) = argmax(region);
                grad[i*2+maxI,j*2+maxJ,k] = delta[i,j,k];


                // const region = convs[i*2..#2, j*2..#2, ..];
                // const maxes = [k_ in 0..#numFilters] max reduce region[..,..,k_];
                // const max = max reduce region [..,..,k];

                // for (i2,j2,k2) in region.domain {
                //     if region[i2,j2,k2] == maxes[k] then {
                //         grad[i*2+i2,j*2+j2,k2] = delta[i,j,k];
                //     }
                // }
            }
            const output = new Tensor(grad);

            // var output = tn.zeros(h,w,numFilters);

            // forall i in 0..#newH with (ref output) {
            //     forall j in 0..#newW with (ref output) {
            //         const region = convs[i*2..#2, j*2..#2, ..];
            //         forall k in 0..#numFilters with (ref output) {
            //             const maxIndex = argmax reduce region[..,..,k];
            //             output[i*2 + maxIndex[0], j*2 + maxIndex[1], k] = delta[i,j,k];
            //         }
            //     }
            // }

            return output;
        }

        proc optimize(mag: real(64)) { }
        proc resetGradients() { }

        proc write(fw: IO.fileWriter) throws {
            // fw.write("[maxpool]");
        }
        proc read(fr: IO.fileReader) throws { }
    }

    record SoftMax {

        var weights: Tensor(2);
        var biases: Tensor(1);

        var weightsGrad: Tensor(2);
        var biasesGrad: Tensor(1);

        proc init(inputLength: int, nodes: int) {
            weights = tn.randn(nodes,inputLength) / inputLength;
            biases = tn.zeros(nodes);

            weightsGrad = tn.zeros(nodes,inputLength);
            biasesGrad = tn.zeros(nodes);
        }

        proc forwardProp(convs: Tensor(3)): Tensor(1) {
            const flattened = convs.flatten();
            const z = (weights * flattened) + biases;
            const exp = tn.exp(z);
            const expSum = + reduce exp.data;
            return exp / expSum;
        }

        proc backward(delta: Tensor(1), convs: Tensor(3)): Tensor(3) {
            const flattened = convs.flatten();
            const Z = (weights * flattened) + biases;
            const exp = tn.exp(Z);
            const expSum = + reduce exp.data;
            const softmax = exp / expSum;
            const dL_dOut = delta;


            var nonZeroIdx: int = -1;
            for i in delta.data.domain do
                if delta[i] != 0 { nonZeroIdx = i; break; }
            
            if nonZeroIdx == -1 then tn.err("Softmax backward: delta is zero vector");
            const i = nonZeroIdx;

            var dOut_dZ: Tensor(1) = (- exp[i]) * (exp / (expSum ** 2.0));
            dOut_dZ[i] = exp[i] * (expSum - exp[i]) / (expSum ** 2.0);
            

            const dZ_dW: Tensor(1) = flattened;
            const dZ_dB: real = 1;
            const dZ_dIn: Tensor(2) = weights;

            const dL_dZ: Tensor(1) = dL_dOut[i] * dOut_dZ;

            const dL_dW: Tensor(2) = dL_dZ * dZ_dW.transpose(); // This should be dL_dW * dL_dZ.transpose();
            const dL_dB: Tensor(1) = dL_dZ * dZ_dB;
            const dL_dIn: Tensor(1) = dZ_dIn.transpose() * dL_dZ; // this is the problem

            // writeln("weights: ", weights.shape);
            // writeln("dL_dW: ", dL_dW.shape);
            weightsGrad += dL_dW; // this might need to be dL_dW.transpose(), along with line 363 alternative
            // writeln("biases: ", biases.shape);
            // writeln("dL_dIn: ", dL_dIn.shape);
            biasesGrad += dL_dB;

            return dL_dIn.reshape((...convs.shape));



            // const (m,n) = weights.shape;
            // const grad: [0..#m, 0..#n] real;

            // forall i in grad.domain {
            //     grad[i,j] = delta[j] * softmax[i];
            // }

            // return grad.reshape(convs.shape);
        }

        proc optimize(mag: real(64)) {
            weights -= mag * weightsGrad;
            biases -= mag * biasesGrad;
        }
        proc resetGradients() {
            weightsGrad.data = 0;
            biasesGrad.data = 0;
        }

        proc write(fw: IO.fileWriter) throws {
            // fw.write("[softmax]");
            weights.write(fw);
            biases.write(fw);
        }
        proc read(fr: IO.fileReader) throws {
            weights.read(fr);
            biases.read(fr);
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
        proc backwardProp(x: Tensor(?), delta: Tensor(?)) {
            return backwardForwardPropHelp(this.layers,0,x,delta);
        }
        proc optimize(mag: real) {
            for param i in 0..#(layers.size) {
                layers[i].optimize(mag);
            }
        }
        proc resetGradients() {
            for param i in 0..#(layers.size) {
                layers[i].resetGradients();
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

        proc save(path: string) throws {
            var file = IO.open(path, IO.ioMode.cw);
            var serializer = new BinaryIO.BinarySerializer(IO.ioendian.big);
            var fw = file.writer(serializer=serializer);
            // fw.write("[network]");
            fw.write(layers.size);
            for param i in 0..#(layers.size) {
                layers[i].write(fw);
            }
            fw.close();
        }

        proc load(path: string) throws {
            var file = IO.open(path, IO.ioMode.rw);
            var deserializer = new BinaryIO.BinaryDeserializer(IO.ioendian.big);
            var fr = file.reader(deserializer=deserializer);
            var size = fr.read(int);
            if size != layers.size then tn.err("Network load: size mismatch");
            for param i in 0..#(layers.size) {
                layers[i].read(fr);
            }
            return this;
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
                new Conv(10),
                new MaxPool(),
                new MaxPool(),
            )
        );
        const image = tn.randn(20,20);
        writeln(image);
        const convs = n2.forwardProp(image);
        writeln(convs);
        // var reversedImage = n2.backwardProp(convs);
        // writeln(reversedImage);
        n2.train([(image,convs)],0.5);

    }
}