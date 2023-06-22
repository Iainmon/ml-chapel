use Map;
use List;
use Random;
use micrograd;


var neuronCounter: int = 0;

class Neuron_ {
    param nin: int;
    var weights: [1..nin] shared Expr;
}
var n_ = new Neuron_(4,[lit("a",1),lit("a",1),lit("a",1),lit("a",1)]);




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
        var sum: Expr = new shared Constant(0); // bias?
        
        for d in dendrites do
            sum = d + sum;
        sum = sum + bias;

        if linear then
            return sum;
        return relu(sum);
    }
    proc updateParams(m: map(string,real)) {
        bias.nudge(m);
        // for w in this.weights do
        //     w.nudge(m);
        weights.nudge(m);
    }

    // proc this(xs: [1..nin] Expr) ref {}

    // proc call(xs: [1..nin] shared Expr) {

    // }
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

        // var neurons = new list(shared Neuron);
        // for i in neuronDomain {
        //     var neuron = new shared Neuron(nin);
        //     neurons.append(neuron);
        // }
        // this.neurons = neurons.toArray();


        // this.neurons = new list(shared Neuron);
        // for i in 1..nout {
        //     var neuron = new shared Neuron(nin);
        //     this.neurons.append(neuron);
        // }
    }

    proc apply(xs: [inputDomain] shared Expr) {

        return this.neurons.apply(xs);

        // var _outs = this.neurons.toArray().apply(xs);
        // var outs = new list(shared Expr);
        // for o in _outs {
        //     outs.append(o);
        // }
        // return outs;
    }
    proc updateParams(m: map(string,real)) {
        this.neruons.updateParams(m);
        // for n in this.neurons {
        //     n.updateParams(m);
        // }
    }
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
        
    }
    proc apply(xs: [inputDomain] shared Expr) {
        var ys = xs;
        for l in this.layers do
            ys = l.apply(ys);

        return ys;
    }

    proc updateParams(m: map(string,real)) {
        this.layers.updateParams(m);
        // for l in this.layers {
        //     l.updateParams(m);
        // }
    }

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
