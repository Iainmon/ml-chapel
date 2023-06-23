use Map;
use List;
use Random;
use micrograd;


proc concat(type t, a: [?da] t, b: [?db] t) {
    var cd = {0..#(a.size + b.size)};
    return [i in cd] if i < a.size then a[i] else b[i - a.size];
    // return [i in cd] if i < a.size then a[a.domain.orderToIndex(i)] else a[a.domain.orderToIndex(i-a.size)];
}

var neuronCounter: int = 0;

class Neuron {
    const nin: int;
    var weightDom: domain(1,int,false);
    var weights: [weightDom] shared Expr;
    var bias: shared Expr;
    var linear: bool = false;

    proc init(const nin: int) {
        this.nin = nin;
        this.weightDom = {0..nin-1};

        var seed = 17 + neuronCounter;
        var rss = new RandomStream(real);

        var weights = new list(shared Expr);

        for i in weightDom {
            var r = rss.getNext(min=-1,max=1);
            var re = lit(neuronCounter:string + "_" + i:string, r);
            weights.append(re);
        }
        this.weights = weights.toArray();

        this.bias = lit(neuronCounter:string + "_bias",rss.getNext(min=-1,max=1));
        neuronCounter += 1;
    }

    proc apply(xs: [weightDom] shared Expr) {
        var dendrites = this.weights * xs;
        var sum: shared Expr = cnst(0); // bias?
        
        for d in dendrites do
            sum = d + sum;
        sum = sum + bias;

        if linear {
            return sum;
        }
        return relu(sum);
    }
    proc updateParams(m: map(string,real)) {
        bias.nudge(m);
        weights.nudge(m);
    }
}


class Layer {
    const nin: int;
    const nout: int;
    var inputDomain: domain(1,int,false);
    var neuronDomain: domain(1,int,false);
    var neurons: [neuronDomain] shared Neuron;

    proc init(const nin: int, const nout: int) {
        this.nin = nin;
        this.nout = nout;
        this.inputDomain = {0..nin-1}; // Same as for each neuron weighDomain
        this.neuronDomain = {0..nout-1};

        this.neurons = [i in 0..nout-1] new shared Neuron(nin);
    }

    proc apply(xs: [inputDomain] shared Expr) do
        return this.neurons.apply(xs);

    proc updateParams(m: map(string,real)) do
        this.neurons.updateParams(m);
}



class Perceptron {
    var sizesDomain: domain(1,int,false);
    var sizes: [sizesDomain] int;

    var layersDomain: domain(1,int,false);
    var layers: [layersDomain] shared Layer;

    var inputDomain: domain(1,int,false);
    // var outputDomain: domain(1,int,false);

    proc init(const nin: int, const nouts: [?D] int) {

        var szs = concat(int,[nin],nouts);

        this.sizesDomain = szs.domain; // {0..nouts.size};// szs.domain;
        this.sizes = szs;
        writeln(szs,szs.domain:string);


        var layers = new list(shared Layer);
        for i in 0..(nouts.size-1) {
            var layer = new shared Layer(this.sizes[i], this.sizes[i + 1]);
            if i == nouts.size {
                for n in layer.neurons do
                    n.linear = true;
            }
            
            layers.append(layer);
        }
        writeln("got here");
        var layersArr = layers.toArray();
        this.layersDomain = layersArr.domain;
        this.layers = layersArr;
        this.inputDomain = this.layers[0].inputDomain; // {0..nin-1};

    }
    proc apply(xs: [inputDomain] shared Expr) {
        var ys = new list(xs);
        for layer in this.layers {
            var zs = layer.apply(ys.toArray());
            ys = new list(zs);
        }
        return ys.toArray();
    }

    proc updateParams(m: map(string,real)) do
        this.layers.updateParams(m);

}

proc cost(result: [?d] shared Expr,expected: [d] shared Expr) {
    var sum = cnst(0);
    var diffs = result - expected;
    for d in diffs {
        sum = sum + (d ** 2);
    }
    return sum;
}

param nin = 2;

var n = new Neuron(nin);
writeln(n);



var inputs = [i in 0..nin-1] cnst(2);


var nout = n.apply(inputs);
writeln(nout.value());


var l = new Layer(nin,2);
var louts = l.apply(inputs);
writeln(louts.value());


var p = new Perceptron(nin,[2,1]);
var pouts = p.apply([cnst(1),cnst(1)]);
writeln(pouts);

/*
var dataset = [
    [cnst(0),cnst(0)],
    [cnst(0),cnst(1)],
    [cnst(1),cnst(0)],
    [cnst(1),cnst(1)]
];
writeln(dataset);




var epochCounter = 1;


var datasetIn = [
    [cnst(0),cnst(0)],
    [cnst(0),cnst(1)],
    [cnst(1),cnst(0)],
    [cnst(1),cnst(1)]
];
var datasetOut = [
    [cnst(0)],
    [cnst(1)],
    [cnst(1)],
    [cnst(0)]
];
var trials = dataset;

proc epoch(mlp: Perceptron) {

    writeln("epoch: ", epochCounter);


    var resultsList = new list([0..#1] shared Expr);
    for d in dataset {
        resultsList.append(mlp.apply(d));
    }
    var results = resultsList.toArray(); // [i in dataset.domain] mlp.apply(dataset[i]);





    var totalCost: Expr = new shared Constant(0);

    for i in results.domain {
        var input = datasetIn[i];
        var expected = datasetOut[i];
        var result = results[i];
        var c = cost(result,expected);
        totalCost = totalCost + c;
    }


    for i in results.domain {
        var input = datasetIn[i];
        var result = results[i];
        var expected = datasetOut[i];
        writeln("Input: ", input.value(), " Expected: ", expected.value(), " Output: ", result.value(), " Local Cost: ", cost(result,expected).value());
    }

    var avgCost = totalCost / cnst(results.size);

    writeln("cost ", avgCost.value());


    var m = makeNudgeMap(avgCost);
    // writeln("delta ", m);

    epochCounter += 1;

    return (avgCost,m);
}


proc train(mlp: shared Perceptron, epochs: int) {
    for ep in 1..epochs {
        var (c,m) = epoch(mlp);
        // c.nudge(m);
        mlp.updateParams(m);
    }
}
var mlp = new shared Perceptron(nin, [4,1]);

train(20000);
*/