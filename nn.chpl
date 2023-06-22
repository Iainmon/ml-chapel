use Map;
use List;
use Random;
use micrograd;


var counter: int = 0;

class Neuron {
    var weights: list(shared Expr);
    var bias: shared Expr;
    var linear: bool = false;

    proc init(nin: int) {
        this.weights = new list(shared Expr);
        var seed = 17;
        var rss = new RandomStream(real,seed);
        for i in 1..nin {
            var r = rss.getNext(min=-1,max=1);
            var re = lit(counter:string + "_" + i:string, r);
            this.weights.append(re);
        }
        this.bias = lit(counter:string + "_bias",0);
        counter += 1;
    }

    proc apply(xs: list(shared Expr)) {
        var dendrites = this.weights.toArray() * xs.toArray(); // + this.bias;
        var sum: Expr = new shared Constant(0);
        for d in dendrites {
            sum = d + sum;
        }
        if linear {
            return sum;
        }
        return relu(sum);
    }
    proc updateParams(m: map(string,real)) {
        // bias.nudge(m);
        for w in this.weights {
            w.nudge(m);
        }
    }

    // proc this(xs: [1..nin] Expr) ref {}
}


class Layer {
    var neurons: list(shared Neuron);
    proc init(nin: int, nout: int) {
        this.neurons = new list(shared Neuron);
        for i in 1..nout {
            var neuron = new shared Neuron(nin);
            this.neurons.append(neuron);
        }
    }

    proc apply(xs: list(shared Expr)) {
        var _outs = this.neurons.toArray().apply(xs);
        var outs = new list(shared Expr);
        for o in _outs {
            outs.append(o);
        }
        return outs;
    }
    proc updateParams(m: map(string,real)) {
        for n in this.neurons {
            n.updateParams(m);
        }
    }
}

class Perceptron {
    var sizes: list(int);
    var layers: list(shared Layer);
    proc init(nin: int, nouts: list(int)) {
        this.sizes = new list(int);
        this.sizes.append(nin);
        this.sizes.append(other=nouts);
        this.layers = new list(shared Layer);
        for i in 0..(nouts.size-1) {
            var l = new shared Layer(this.sizes[i], this.sizes[i + 1]);
            if i == nouts.size {
                for n in l.neurons {
                    n.linear = true;
                }
            }
            this.layers.append(l);
        }
    }
    proc apply(xs: list(shared Expr)) {
        var ys = xs;
        for l in this.layers.these() {
            ys = l.apply(ys);
        }
        return ys;
    }

    proc updateParams(m: map(string,real)) {
        for l in this.layers {
            l.updateParams(m);
        }
    }

}

const nin = 2;

var n = new Neuron(nin);
writeln(n);

var inputs = new list(shared Expr);
for i in 1..nin {
    // inputs.append(lit("in" + i:string, 2));
    inputs.append((new shared Constant(2)): Expr);
}

var nout = n.apply(inputs);
writeln(nout.value());

var l = new Layer(nin,2);
var louts = l.apply(inputs);
writeln(louts.toArray().value());


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

var mlp = new Perceptron(nin, [4,2,1]:list(int));

proc train(epochs: int) {
    for ep in 1..epochs {
        var (c,m) = epoch(mlp);
        // c.nudge(m);
        mlp.updateParams(m);
    }
}

train(5000);

// class Neuron {
//     var nin: int;
// }