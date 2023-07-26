
record Tensor {
    param rank: int;
    type eltType = real;

    var _domain: domain(rank,int);
    var data: [_domain] eltType;

    proc init(type eltType, shape: int ...?dim) {
        this.rank = dim ;
        this.eltType = eltType;
        var ranges: dim*range;
        for (size,r) in zip(shape,ranges) do r = 0..#size;
        this._domain = {(...ranges)};
    }
}

record Dense {

    proc forwardProp(x: Tensor(1)): Tensor(1) {
        return new Tensor(real, 0);
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
    const layers;

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
            new Conv(),
            new MaxPool(),
            new Dense(),
        )
    );

    var p = n.forwardProp(new Tensor(real, 0, 0));

    writeln(p.type:string);
}
