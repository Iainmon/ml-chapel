
module Chai {
    
    import Tensor as tn;
    use Tensor;

    import IO;
    import BinaryIO;
    import Reflection;    

    record Dense {

        var outputSize: int;

        var bias: Tensor(1);
        var weights: Tensor(2);

        var biasGrad: Tensor(1);
        var weightsGrad: Tensor(2);

        var uninitialized = true;

        type gradientType = (Tensor(2),Tensor(1));


        proc init(outputSize: int) {
            this.outputSize = outputSize;

            bias = new Tensor(1,real); // tn.randn(outputSize);
            weights = new Tensor(2,real); // tn.randn(outputSize, inputSize);

            biasGrad = new Tensor(1,real);
            weightsGrad = new Tensor(2,real);
        }

        proc forwardProp(input: Tensor(1)): Tensor(1) {
            if uninitialized {
                const inputSize = * reduce input.shape;
                const stddevB = sqrt(2.0 / outputSize);
                const stddevW = sqrt(2.0 / (inputSize + outputSize));
                bias = tn.zeros(outputSize); // tn.randn(outputSize,mu=0.0,sigma=stddevB);
                weights = tn.randn(outputSize, inputSize,mu=0.0,sigma=stddevW);

                biasGrad = tn.zeros(outputSize);
                weightsGrad = tn.zeros(outputSize, inputSize);
                uninitialized = false;
            }
            const activation = (weights * input) + bias;
            return activation;
        }

        proc initialGradient(): this.gradientType {
            return (tn.zeros((...weights.shape)),tn.zeros((...bias.shape)));
        }

        proc backward(delta: Tensor(1), input: Tensor(1)): Tensor(1) {
            const newDelta = weights.transpose() * delta;
            biasGrad    += newDelta;
            weightsGrad += newDelta * input.transpose();
            return newDelta;
        }
        proc backward(delta: Tensor(1), input: Tensor(1), ref myGradient: this.gradientType): Tensor(1) {
            const newDelta = weights.transpose() * delta;

            myGradient[0] += newDelta * input.transpose();
            myGradient[1] += newDelta;

            return newDelta;
        }

        proc optimize(mag: real(64)) {
            bias -= mag * biasGrad;
            weights -= mag * weightsGrad;
        }

        proc optimize(mag: real, ref myGradient: this.gradientType) {
            bias -= mag * myGradient[1];
            weights -= mag * myGradient[0];
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
            uninitialized = false;
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
        var filters: Tensor(4);
        var filtersGrad: Tensor(4);
        var isFirstLayer = false;
        var stride: int = 1;
        var padding: int = 0;

        type gradientType = Tensor(4);

        proc init(inChannels: int,outChannels: int, kernelSize: int = 3, stride: int = 1, padding: int = 0) {
            const numFilters = outChannels;
            this.numFilters = numFilters;

            const fanIn = kernelSize * kernelSize * inChannels;
            const fanOut = outChannels;
            const stddev = sqrt(2.0 / (fanOut + fanIn));
            this.filters = tn.randn(outChannels,kernelSize,kernelSize,inChannels,mu=0.0,sigma=stddev);
            // this.filters = tn.randn(numFilters,kernelSize,kernelSize,inChannels) / (kernelSize:real ** 2.0);
            
            this.filtersGrad = tn.zeros(numFilters,kernelSize,kernelSize,inChannels);
            this.stride = stride;
            this.padding = padding;
        }

        proc initialGradient(): this.gradientType {
            return tn.zeros((...filters.shape));
        }

        proc forwardProp(images: Tensor(3)): Tensor(3) {
            const (h,w,channels) = images.shape;
            const (outChannels,kh,kw,inChannels) = filters.shape;

            if channels != inChannels {
                writeln("input: ", images.shape);
                writeln("filters: ", filters.shape);
                tn.err("Conv forwardProp: inChannels mismatch");
            }

            var convs = new Tensor(3,real);
            const (newH,newW) = correlateShape((kh,kw),(h,w),stride,padding);
            convs.reshapeDomain({0..#newH, 0..#newW, 0..#outChannels});
            forall f in 0..#outChannels with (ref convs, var filter: Tensor(3) = tn.zeros(kh,kw,channels)) {
                filter.data = filters[f,..,..,..];
                convs[..,..,f] = correlate(filter=filter,input=images,stride=stride,padding=padding);
            }
            convs.data /= (inChannels:real);
            return convs;
            
        }
    

        proc backward(delta: Tensor(3), images: Tensor(3)): Tensor(3) {
            const (h,w,channels) = images.shape;
            const (outChannels,kh,kw,inChannels) = filters.shape;
            const (dh,dw,dc) = delta.shape;


            if dc != outChannels then tn.err("Conv backward: outChannels mismatch");
            if channels != inChannels then tn.err("Conv backward: inChannels mismatch");

            const dL_dF = tn.filterGradient(images,delta,stride,padding,kh);
            filtersGrad += dL_dF;

            var dL_dX = new Tensor(3,real);
            dL_dX.reshapeDomain({0..#h, 0..#w, 0..#inChannels});
            forall (m,n,ci) in {0..#h, 0..#w, 0..#inChannels} with (ref dL_dX) {
                var sum = 0.0;
                forall co in 0..#outChannels with (+ reduce sum) {
                    forall (i,j) in {0..#dh, 0..#dw} with (+ reduce sum) {
                        const (dXi,dXj) = correlateWeightIdx((kh,kw),(m,n),(i,j),stride,padding);
                        if dXi != -1 then
                            sum += delta[i,j,co] * filters[co,dXi,dXj,ci];
                    }
                }
                dL_dX[m,n,ci] = sum;
            }
            return dL_dX;
        }
        
        proc backward(delta: Tensor(3), images: Tensor(3),ref myGradient: this.gradientType): Tensor(3) {
            const (h,w,channels) = images.shape;
            const (outChannels,kh,kw,inChannels) = filters.shape;
            const (dh,dw,dc) = delta.shape;


            if dc != outChannels then tn.err("Conv backward: outChannels mismatch");
            if channels != inChannels then tn.err("Conv backward: inChannels mismatch");

            const dL_dF = tn.filterGradient(images,delta,stride,padding,kh);
            myGradient += dL_dF;

            var dL_dX = new Tensor(3,real);
            dL_dX.reshapeDomain({0..#h, 0..#w, 0..#inChannels});
            forall (m,n,ci) in {0..#h, 0..#w, 0..#inChannels} with (ref dL_dX) {
                var sum = 0.0;
                forall co in 0..#outChannels with (+ reduce sum) {
                    forall (i,j) in {0..#dh, 0..#dw} with (+ reduce sum) {
                        const (dXi,dXj) = correlateWeightIdx((kh,kw),(m,n),(i,j),stride,padding);
                        if dXi != -1 then
                            sum += delta[i,j,co] * filters[co,dXi,dXj,ci];
                    }
                }
                dL_dX[m,n,ci] = sum;
            }
            return dL_dX;
        }
        
        proc optimize(mag: real(64)) {
            filters -= mag * filtersGrad;
        }
        proc optimize(mag: real, ref myGradient: this.gradientType) {
            filters -= mag * myGradient;
        }

        proc resetGradients() {
            filtersGrad.data = 0.0;
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

        type gradientType = nothing;

        proc initialGradient(): this.gradientType {
            return none;
        }

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

            return output;
        }
        proc backward(delta: Tensor(3), convs: Tensor(3), ref myGradient: this.gradientType): Tensor(3) {
            const (h,w,numFilters) = convs.shape;

            const newH: int = h / 2;
            const newW: int = w / 2;

            var grad: [0..#h, 0..#w, 0..#numFilters] real;
            forall (i,j,k) in delta.data.domain {
                const region = convs[i*2..#2, j*2..#2, k];
                const (maxI,maxJ) = argmax(region);
                grad[i*2+maxI,j*2+maxJ,k] = delta[i,j,k];
            }
            const output = new Tensor(grad);

            return output;
        }

        proc optimize(mag: real(64)) { }
        proc optimize(mag: real, ref myGradient: this.gradientType) { }
        proc resetGradients() { }

        proc write(fw: IO.fileWriter) throws {
            // fw.write("[maxpool]");
        }
        proc read(fr: IO.fileReader) throws { }
    }

    operator +(x: nothing, y: nothing) {
        return none;
    }
    operator +=(ref x: nothing, y: nothing) {}

    record ReLU {
        var a: real = 0.0;
        proc init(a: real = 0.0) { this.a = a; }
        proc forwardProp(input: Tensor(?rank)) {
            var output = new Tensor(rank,real);
            output.reshapeDomain(input.domain);
            foreach i in output.domain {
                const y = input.data[i];
                output.data[i] = max(y,a * y);
            }
            return output;
        }
        proc backward(delta: Tensor(?rank),input: Tensor(rank)) {
            var output = new Tensor(rank,real);
            output.reshapeDomain(input.domain);
            foreach i in output.domain {
                const y = input.data[i];
                const dy = delta.data[i];
                output.data[i] = if y > 0.0 then dy else a * dy;
            }
            return output;
        }
        proc optimize(mag: real(64)) { }
        proc resetGradients() { }
        proc write(fw: IO.fileWriter) throws { }
        proc read(fr: IO.fileReader) throws { }
    }

    record Flatten {
        proc init() { }
        proc forwardProp(input: Tensor(?inRank)): Tensor(1) {
            return input.flatten();
        }
        proc backward(delta: Tensor(1), input: Tensor(?inRank)): Tensor(inRank) {
            return delta.reshape(input.domain);
        }
        proc optimize(mag: real(64)) { }
        proc resetGradients() { }
        proc write(fw: IO.fileWriter) throws { }
        proc read(fr: IO.fileReader) throws { }
    }

    record SoftMax {

        var weights: Tensor(2);
        var biases: Tensor(1);

        var weightsGrad: Tensor(2);
        var biasesGrad: Tensor(1);

        var uninitialized: bool = true;
        var outputSize: int = 0;

        type gradientType = (Tensor(2),Tensor(1));

        proc init(inputLength: int, nodes: int) {
            weights = tn.randn(nodes,inputLength);// / inputLength;
            biases = tn.randn(nodes);

            weightsGrad = tn.zeros(nodes,inputLength);
            biasesGrad = tn.zeros(nodes);
            uninitialized = false;
        }

        proc init(outputSize: int) {
            this.outputSize = outputSize;
        }

        proc initialGradient(): this.gradientType {
            return (tn.zeros((...weights.shape)),tn.zeros((...biases.shape)));
        }


        proc forwardProp(input: Tensor(?)): Tensor(1) {
            tn.debugWrite("[enter softmax forward]");

            if uninitialized {
                const inputSize = * reduce input.shape;
                if inputSize < 1 then tn.err("Softmax input size must be > 0");

                const stddev = sqrt(2.0 / (inputSize + outputSize));
                // weights = tn.randn(outputSize,inputSize,mu=0.0,sigma=stddev);
                weights = tn.randn(outputSize,inputSize) / (inputSize: real);
                biases = tn.zeros(outputSize);// tn.randn(outputSize) / (outputSize: real);

                weightsGrad = tn.zeros(outputSize,inputSize);
                biasesGrad = tn.zeros(outputSize);

                uninitialized = false;
            }

            const flattened = input.flatten();
            const z = (weights * flattened) + biases;
            return tn.softmax(z);
        }

        proc backward(delta: Tensor(1), input: Tensor(?outRank)): Tensor(outRank) {
            const flattened = input.flatten();
            const Z = (weights * flattened) + biases;
            const (exp,expSum,softmax) = tn.softmaxParts(Z);
            // const exp = tn.exp(Z);
            // const expSum = + reduce exp.data;
            // const softmax = exp / expSum;
            const dL_dOut = delta;


            var nonZeroIdx: int = -1;
            for i in delta.data.domain do
                if delta[i] != 0.0 { nonZeroIdx = i; break; }
            
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

            return dL_dIn.reshape(input.domain);
        }

        proc backward(delta: Tensor(1), input: Tensor(?outRank), ref myGradient: this.gradientType): Tensor(outRank) {
            const flattened = input.flatten();
            const Z = (weights * flattened) + biases;
            const (exp,expSum,softmax) = tn.softmaxParts(Z);
            // const exp = tn.exp(Z);
            // const expSum = + reduce exp.data;
            // const softmax = exp / expSum;
            const dL_dOut = delta;


            var nonZeroIdx: int = -1;
            for i in delta.data.domain do
                if delta[i] != 0.0 { nonZeroIdx = i; break; }
            
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


            myGradient[0] += dL_dW; // this might need to be dL_dW.transpose(), along with line 363 alternative
            myGradient[1] += dL_dB;

            return dL_dIn.reshape(input.domain);
        }

        proc optimize(mag: real(64)) {
            weights.data -= mag * weightsGrad.data;
            biases.data -= mag * biasesGrad.data;
        }
        proc optimize(mag: real, ref myGradient: this.gradientType) {
            weights.data -= mag * myGradient[0].data;
            biases.data -= mag * myGradient[1].data;
        }
        proc resetGradients() {
            weightsGrad.data = 0.0;
            biasesGrad.data = 0.0;
        }

        proc write(fw: IO.fileWriter) throws {
            // fw.write("[softmax]");
            weights.write(fw);
            biases.write(fw);
        }
        proc read(fr: IO.fileReader) throws {
            weights.read(fr);
            biases.read(fr);
            uninitialized = false;
        }

    }


    proc forwardPropHelp(ref layers, param n: int, x: Tensor(?)) {
        if n == layers.size then return x;

        // const xNext = layers[n].forwardProp(x);
        // return forwardPropHelp(layers, n+1, xNext);

        // Better? 
        return forwardPropHelp(layers, n+1, layers[n].forwardProp(x));
    }

    proc backwardPropHelp(ref layers, param n: int, x: Tensor(?)) {
        if n == 0 then return layers[0].backward(x);

        const xNext = layers[n].backward(x);
        return backwardPropHelp(layers, n-1, xNext);
    }

    proc backwardForwardPropHelp(ref layers, param n: int,x : Tensor(?), lastDelta: Tensor(?)) {

        if n == layers.size - 1 {
            return layers[n].backward(lastDelta,x);
        }

        // const lastInput = layers[n].forwardProp(x);
        // const delta = backwardForwardPropHelp(layers, n+1, lastInput, lastDelta);
        // return layers[n].backward(delta,x);

        // Better? 
        return layers[n].backward(backwardForwardPropHelp(layers, n + 1, layers[n].forwardProp(x), lastDelta),x);
    }

    proc backwardForwardPropHelpGradRef(ref layers,ref layerGrads, param n: int,x : Tensor(?), lastDelta: Tensor(?)) {
        ref layerGradient: layers[n].gradientType = layerGrads[n];

        if n == layers.size - 1 {
            return layers[n].backward(lastDelta,x,layerGradient);
        }

        // const lastInput = layers[n].forwardProp(x);
        // const delta = backwardForwardPropHelp(layers, n+1, lastInput, lastDelta);
        // return layers[n].backward(delta,x);

        // Better? 
        const lastInput = layers[n].forwardProp(x);
        const delta = backwardForwardPropHelpGradRef(layers,layerGrads, n + 1, lastInput, lastDelta);
        return layers[n].backward(delta,x,layerGradient);
    }

    proc tupleTypeBuilder(ref layers, param n: int) {
        if n == layers.size - 1 then return (layers[n].gradientType,);
        return (layers[n].gradientType, (...tupleTypeBuilder(layers,n+1)));
    }

    proc convert(args) type where isTuple(args) {
        proc rec(param idx: int) type {
            if idx == args.size - 1 then return (args[idx].gradientType,);
            return (args[idx].gradientType, (...rec(idx+1)));
        }
        return rec(0);
    }

    record Network {
        var layers;

        proc gradientType type {
            // return (...tupleTypeBuilder(layers,0));
            return convert(layers);
        }

        proc init(layers) {
            this.layers = layers;
            if Reflection.hasField(this.layers[0].type, "isFirstLayer") {
                this.layers[0].isFirstLayer = true;
            }
        }

        proc initialGradient() {
            var grads: this.gradientType;
            for param i in 0..#(layers.size) {
                grads[i] = layers[i].initialGradient();
            }
            return grads;
        }

        proc forwardProp(x: Tensor(?)) {
            return forwardPropHelp(this.layers, 0, x);
        }

        proc backwardProp(x: Tensor(?)) {
            writeln("dont use me, Network.backwardProp");
            return backwardPropHelp(this.layers,this.layers.size - 1,x);
        }
        proc backwardProp(x: Tensor(?), delta: Tensor(?)) {
            return backwardForwardPropHelp(this.layers,0,x,delta);
        }

        proc backwardProp(x: Tensor(?), delta: Tensor(?), ref layerGrads: this.gradientType) {
            return backwardForwardPropHelpGradRef(layers,layerGrads,0,x,delta);
        }

        proc optimize(mag: real, ref layerGrads: this.gradientType) {
            for param i in 0..#(layers.size) {
                ref layerGradient = layerGrads[i];
                layers[i].optimize(mag,layerGradient);
            }
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

    /* Implements + reduction over numeric data. */
    class PlusReduceOp: ReduceScanOp {

        /* the type of the elements to be reduced */
        type eltType;

        /* task-private accumulator/reduction state */
        var value: eltType;

        /* identity w.r.t. the reduction operation */
        proc identity { return value: eltType; }

        /* accumulate a single element onto the accumulator */
        proc accumulate(elm)  { value = value + elm; }

        /* accumulate a single element onto the state */
        proc accumulateOntoState(ref state, elm)  { state = state + elm; }

        /* accumulate the value of the outer variable at the entry to the loop */
        // Note: this method is optional. If it is not provided,
        // accumulate(outerVar) is used instead.
        proc initialAccumulate(outerVar) { value = value + outerVar: eltType; }

        // Note: 'this' can be accessed by multiple calls to combine()
        // concurrently. The Chapel implementation serializes such calls
        // with a lock on 'this'.
        // 'other' will not be accessed concurrently.
        /* combine the accumulations in 'this' and 'other' */
        proc combine(other: borrowed PlusReduceOp)   { value = value + other.value; }

        /* Convert the accumulation into the value of the reduction
            that is reported to the user. This is trivial in our case. */
        proc generate() { return value; }

        /* produce a new instance of this class */
        proc clone() { return new unmanaged PlusReduceOp(eltType=eltType,value=value); }
    }


    proc main() {
        // var n = new Network(
        //     (
        //         new Dense(3,3),
        //         new Sigmoid(3),
        //         new Dense(3,6),
        //         new Sigmoid(6)
        //     )
        // );

        // const inv: [0..#3] real = [1,2,3];
        // const input = new Tensor(inv);

        // var output = n.forwardProp(input);
        // var reversedInput = n.backwardProp(output);
        
        // writeln(input);
        // writeln(output);
        // writeln(reversedInput);

        // const t = tn.randn(3,4);
        // writeln(t);

        // var shape = (3,4,5);
        // for i in 0..<(3 * 4 * 5) {
        //     writeln(i, " ", tn.nbase(shape,i));
        // }

        var n2 = new Network(
            (
                new Conv(1,8,3),
                new MaxPool(),
                new Conv(8,12,3),
                new MaxPool(),
                new SoftMax(5 * 5 * 12, 10)
            )
        );
        const image = tn.randn(28,28,1);
        writeln(image);
        const convs = n2.forwardProp(image);
        writeln(convs);
        // var reversedImage = n2.backwardProp(convs);
        // writeln(reversedImage);
        // n2.train([(image,convs)],0.5);

    }
}