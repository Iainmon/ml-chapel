use Map;
use List;
use Random;
use micrograd;


var neuronCounter: int = 0;

class Neuron {
    param nin: int;
    var weightDom: domain(1,int,false);
    var weights: [weightDom] shared Expr;
    var bias: shared Expr;
    var linear: bool = false;

    proc init(param nin: int) {
        this.nin = nin;
        this.weightDom = {0..nin-1};

        var seed = 17;
        var rss = new RandomStream(real,seed);

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
        var sum: Expr = cnst(0); // bias?
        
        for d in dendrites do
            sum = d + sum;
        sum = sum + bias;

        if linear then
            return sum;
        return relu(sum);
    }
    proc updateParams(m: map(string,real)) {
        bias.nudge(m);
        weights.nudge(m);
    }
}


class Layer {
    param nin: int;
    param nout: int;
    var inputDomain: domain(1,int,false);
    var neuronDomain: domain(1,int,false);
    var neurons: [neuronDomain] shared Neuron(nin);

    proc init(param nin: int, param nout: int) {
        this.nin = nin;
        this.nout = nout;
        this.inputDomain = {0..nin-1}; // Same as for each neuron weighDomain
        this.neuronDomain = {0..nout-1};

        this.neurons = [i in 0..nout-1] new shared Neuron(nin);
    }

    proc apply(xs: [inputDomain] shared Expr) do
        return this.neurons.apply(xs);

    proc updateParams(m: map(string,real)) do
        this.neruons.updateParams(m);
}

class Perceptron {
    var sizesDomain: domain(1,int,false);
    var sizes: [sizesDomain] int;
    var layers: [sizesDomain] shared Layer;
    var inputDomain: domain(1,int,false);
    // var outputDomain: domain(1,int,false);

    proc init(param nin: int, param nouts: [?D] int) {
        var szs = [nin] + nouts;
        this.sizeDomain = szs.domain;
        this.sizes = szs;


        var layers = new list(shared Layer);
        for i in 0..(nouts.size-1) {
            var layer = new shared Layer(this.sizes[i], this.sizes[i + 1]);
            if i == nouts.size then
                layer.neruons.linear = true;
            
            layers.append(layer);
        }
        this.layers = layers.toArray();
        this.inputDomain = this.layers[0].inputDomain; // {0..nin-1};

        // this.sizes.append(nin);
        // this.sizes.append(other=nouts);
        // this.layers = new list(shared Layer);
        // for i in 0..(nouts.size-1) {
        //     var l = new shared Layer(this.sizes[i], this.sizes[i + 1]);
        //     if i == nouts.size {
        //         for n in l.neurons {
        //             n.linear = true;
        //         }
        //     }
        //     this.layers.append(l);
        // }
    }
    proc apply(xs: [inputDomain] shared Expr) {
        var ys = xs;
        for l in this.layers do
            ys = l.apply(ys);
        return ys;
    }

    proc updateParams(m: map(string,real)) do
        this.layers.updateParams(m);

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




// not yet ported
/*


proc cost(result: list(shared Expr),expected: list(shared Expr)) {
    var sum = new shared Constant(0): Expr;
    var diffs = result.toArray() - expected.toArray();
    for d in diffs {
        sum = sum + (d ** 2);
    }
    return sum;
}

var epochCounter = 1;

proc epoch(mlp: Perceptron) {

    writeln("epoch: ", epochCounter);
    var trials = new list(list(shared Expr));

    for i in 0..1 {
        for j in 0..1 {
            var trial = new list(shared Expr);
            trial.append(new shared Constant(i): Expr);
            trial.append(new shared Constant(j): Expr);
            trials.append(trial);
        }
    }

    // writeln(trials);


    var trialResults = new list(list(shared Expr));
    for t in trials {
        var result = mlp.apply(t);
        trialResults.append(result);
        var res = + reduce t.toArray().value();
        var expect = 0;
        if res == 1 {
            expect = 1;
        }
        writeln("Input: ", t.toArray().value(), " Expected: ", expect, " Output: ", result.toArray().value());
    }

    var totalCost: Expr = new shared Constant(0);

    var xorCounter = 0;
    for rs in trialResults {
        var es = new list(shared Expr);
        var n = 0;
        n = (+ reduce trials[xorCounter].toArray().value()): int;
        if n != 1 { n = 0; }
        // if xorCounter == 1 || xorCounter == 2 {
        //     n = 1;
        // }
        xorCounter += 1;
        es.append(new shared Constant(n): Expr);

        var c = cost(rs,es);
        totalCost = totalCost + c;
    }

    var avgCost = totalCost / (new shared Constant(trialResults.size): Expr);

    writeln("cost ", avgCost.value());

    // avgCost.showGradient("cost");

    var m = makeNudgeMap(avgCost);
    writeln("delta ", m);

    epochCounter += 1;

    return (avgCost,m);
}

var mlp = new Perceptron(nin, [2,1]:list(int));

proc train(epochs: int) {
    for ep in 1..epochs {
        var (c,m) = epoch(mlp);
        // c.nudge(m);
        mlp.updateParams(m);
    }
}

train(1000);

*/