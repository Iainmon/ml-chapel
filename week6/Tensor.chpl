
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
    proc init(shape: int ...?dim) {
        this.rank = dim ;
        this.eltType = real;
        var ranges: dim*range;
        for (size,r) in zip(shape,ranges) do r = 0..#size;
        this._domain = {(...ranges)};
    }
    // proc init(shape: int ...?d) {
    //     this.rank = d;
    //     this.eltType = real;
    //     this._domain = {(...shape)};
    //     this.data = 0.0;
    // }

    // forwarding data only this;
}

class Layer {
    // proc forwardProp(x: Tensor(?)): Tensor(?) { return x; }
    proc dimIn() param: int do return 1;
    proc dimOut() param: int do return 1;
    // proc forwardProp(x: Tensor(?din)): Tensor(?dout) where dout == this.dimOut() && din == this.dimIn() {
    //     new Tensor(this.dimOut());
    // }

    proc forwardProp(param din: int, param dout: int, x: Tensor(din)): Tensor(dout) {
        return new Tensor(dout);
    }

}

class Dense: Layer {
    override proc dimIn() param: int do return 1;
    override proc dimOut() param: int do return 1;
    override proc forwardProp(param din: int, param dout: int, x: Tensor(din)): Tensor(dout) where din == 1 && dout == 1 {
        return x;
    }
    // override proc forwardProp(param din: int, param dout: int, x: Tensor(din)): Tensor(dout) where din == 2 && dout == 1 {
    //     return new Tensor(1);
    // }
}


class Conv: Layer {
    override proc dimIn() param: int do return 2;
    override proc dimOut() param: int do return 2;
    proc forwardProp(param din: int, param dout: int, x: Tensor(din)): Tensor(dout) where din == 2 && dout == 2 {
        return x;
    }
}

class MaxPool: Layer {
    override proc dimIn() param: int do return 2;
    override proc dimOut() param: int do return 1;
    proc forwardProp(param din: int, param dout: int, x: Tensor(din)): Tensor(dout) where din == 2 && dout == 1 {
        return new Tensor(real,2,2);
    }
}

proc feedForwardRec(x: Tensor(?din), layers: [] shared Layer): Tensor(layers.first.dimOut()) {
    const l = layers.first;
    param dout = l.dimOut();
    const y = l.forwardProp(din,dout,x);
    if layers.size == 1 then return y;
    return feedForwardRec(y,layers[1..]);
}

class Network {
    var _layersDomain: domain(1,int);
    var layers: [_layersDomain] shared Layer;

    proc init(layers ...?nl) {
        this._layersDomain = {0..#nl};
        this.layers = layers;
    }
    proc forwardProp(x: Tensor(?)): Tensor(?) {

        var y1: Tensor(1);
        var y2: Tensor(2);
        var lastRank: int = selectAssign(x,y1,y2);

        for l in layers {
            param din = l.dimIn();
            param dout = l.dimOut();
            select lastRank {
                when 1 {
                    var z = l.forwardProp(din,dout,y1);
                    lastRank = selectAssign(z,y1,y2);
                }
                when 2 {
                    var z = l.forwardProp(2,dout,y2);
                    lastRank = selectAssign(z,y1,y2);
                }
            }
        }
        select lastRank {
            when 1 do return y1;
            when 2 do return y2;
        }
        halt("unreachable");

    }
    proc selectAssign(x: Tensor(?), ref y1: Tensor(1), ref y2: Tensor(2)): int {
        select x.rank {
            when 1 {
                y1 = x;
                return 1;
            }
            when 2 {
                y2 = x;
                return 2;
            }
        }
    }
    // proc forwardProp(x: Tensor(?)): Tensor(?) {
    //     var y = x;
    //     for l in this.layers {
    //         y = l.forwardProp(y);
    //     }
    //     return y;
    // }
}

proc main() {
    var net = new Network(
                // new shared Dense(),
                // new shared MaxPool(),
                new shared Dense()
                );
    // net.forwardProp(new Tensor(real,2));
    feedForwardRec(new Tensor(real,2),net.layers);
}




/*
record NDArray {
    type eltType;
    var _shapeDomain: domain(1,int);
    var shape: [_shapeDomain] int;

    var _dataDomain: domain(1,int);
    var data: [_dataDomain] eltType;

    proc init(type eltType, shape: [?d] int) {
        this.eltType = eltType;
        this._shapeDomain = d;
        this.shape = shape;
        const size = * reduce shape;
        this._dataDomain = {0..#size};
        
    }

    proc this(is: int ...?n) {
        var idxs: [0..#n] int;
        for i in 0..#(n - 1) {
            idxs[i] = is[i] * (* reduce this.shape[i+1..]);
        }
        idxs[n - 1] = is[n - 1];
        return this.data[idxs];
    }
}



class TensorClass {
    type eltType;
    var array: NDArray(eltType);


}

type Tensor = shared TensorClass(?);








// var a = new NDArray(int,[5,5]);
// writeln(a[1,1]);
*/