

use micrograd;
use List;

proc fun(xs: [?D] shared Expr) {
    for i in D {
        writeln(i, ":", xs[i].value());
    }
}

var exprArr = [lit("a",1),lit("b",2)];

fun(exprArr);


class Neuron {
    param nin: int;
    var weightDom: domain(1,int,false);
    var weights: [weightDom] shared Expr;

    proc init(param nin: int, weights: [?D] shared Expr) {
        this.nin = nin;
        this.weightDom = D;
        this.weights = weights;
    }

    proc apply(xs: [weightDom] shared Expr) {
        // var vs = xs.values();
        // var vs_ = (new list(xs)).values();

        var ints = [1,2,3,4,5];
        // var squares = for i in ints do i * i;// [i * i for i in ints];
        var squares = [i in ints] i * i;
        var sums = + reduce squares;
        var sums = sum(xs, new shared Constant(0));
        var exprSums = + reduce xs;

        return exprSums;
        // return new list(xs);
    }

    
}

const exprs: [0..3] shared Expr = [lit("a",1),lit("a",1),lit("a",1),lit("a",1)];

var n = new Neuron(4,[lit("a",1),lit("a",1),lit("a",1),lit("a",1)]);

var l = n.apply(exprs);
writeln(l);





