class Cow {
    var age: int;
}

class ByteStream {
    var cow: Cow;
    proc init(in c: Cow) {
        cow = c;
        writeln(cow);
    }
}

var bs = new ByteStream(new Cow(10));

var bs2 = new ByteStream(new shared Cow(10));

type SharedCow = shared Cow;

var bs3 = new ByteStream(new SharedCow(10));

record Tensor {
    param rank: int;
    type eltType;

    var _domain: domain(rank);
    var data: [_domain] eltType;

    proc init(type eltType, shape: int ...?dim) {

        this.rank = dim;
        this.eltType = eltType;
        var ranges: dim*range;
        // var dims: [0..#dim] int;
        for (size,r) in zip(shape,ranges) do r = 0..#size;
        this._domain = {(...ranges)};
    }
}
var t = new Tensor(real,2,3); // Should be a 2x3 matrix

writeln(t.rank);