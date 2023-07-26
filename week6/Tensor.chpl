module Tensor {

    import Linear as la;
    import Math;

    proc err(args...?n) {
        var s = "";
        for param i in 0..<n {
            s += args(i): string;
        }
        try! throw new Error(s);
    }
    
    record Tensor {
        param rank: int;
        type eltType = real;

        var _domain: domain(rank,int);
        var data: [_domain] eltType;

        proc shape do return this._domain.shape;

        proc init(type eltType, shape: int ...?dim) {
            this.rank = dim ;
            this.eltType = eltType;
            var ranges: dim*range;
            for (size,r) in zip(shape,ranges) do r = 0..#size;
            this._domain = {(...ranges)};
        }
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
        proc init(param rank: int, type eltType) {
            this.rank = rank;
            this.eltType = eltType;
            var ranges: rank*range;
            for r in ranges do r = 0..#0;
            this._domain = {(...ranges)};
        }
        proc init(data: [?d] ?eltType) {
            this.rank = d.rank;
            this.eltType = eltType;
            this._domain = d;
            this.data = data;
        }
        // proc init(shape: int ...?d) {
        //     this.rank = d;
        //     this.eltType = real;
        //     this._domain = {(...shape)};
        //     this.data = 0.0;
        // }

        // forwarding data only this;

        
        proc fmap(fn) {
            const data = fn(this.data);
            return new Tensor(data);
        }
        

    }


    operator +(lhs: Tensor(?d), rhs: Tensor(d)) {
        const data = lhs.data + rhs.data;
        return new Tensor(data);
    }
    operator +=(ref lhs: Tensor(?d), const ref rhs: Tensor(d)) {
        lhs.data += rhs.data;
    }
    operator -(lhs: Tensor(?d), rhs: Tensor(d)) {
        const data = lhs.data - rhs.data;
        return new Tensor(data);
    }
    operator -=(ref lhs: Tensor(?d), const ref rhs: Tensor(d)) {
        lhs.data -= rhs.data;
    }
    operator *(lhs: Tensor(?d), rhs: Tensor(d)) {
        // Hermitian product, not composition
        const data = lhs.data * rhs.data;
        return new Tensor(data);
    }
    operator *=(ref lhs: Tensor(?d), const ref rhs: Tensor(d)) {
        lhs.data *= rhs.data;
    }

    operator *(lhs: Tensor(2,?eltType), rhs: Tensor(1,eltType)): Tensor(1,eltType) {
        const (m,n) = lhs.shape;
        const (p,) = rhs.shape;
        if n != p then
            err("Trying to apply a matrix of shape ",lhs.shape, " to a vector of shape ", rhs.shape);
        const a = lhs.data;
        const v = rhs.data;
        var w: [0..#m] eltType;
        forall i in 0..#m {
            const row = a[i,..];
            w[i] = + reduce (row * v);
        }
        return new Tensor(w);
    }
    operator *(lhs: Tensor(1,?eltType), rhs: Tensor(2,eltType)): Tensor(1,eltType) {
        const (p,) = lhs.shape;
        const (m,n) = rhs.shape;
        if m != 1 then
            err("Trying to apply a vector of shape ",lhs.shape, " to a matrix of shape ", rhs.shape);
        
        const a = lhs.data;
        const v = rhs.data;
        var b: [0..#p, 0..#n] eltType;
        forall (i,j) in b.domain {
            b[i,j] = v[i] * a[0,j];
        }
        return new Tensor(b);
    }

    proc matToTens(m: la.Matrix(?t)): Tensor(2,t) {
        return new Tensor(m.underlyingMatrix);
    }
    proc vecToTens(v: la.Vector(?t)): Tensor(1,t) {
        return new Tensor(v.underlyingVector);
    }

    proc randn(shape: int ...?d): Tensor(d,real) {
        var t = new Tensor((...shape));
        var m: [t.data.domain] real;
        forall i in m.domain {
            var x: real = la.normal();
            m[i] = x;
        }
        return new Tensor(m);
    }
    proc zeros(shape: int ...?d): Tensor(d,real) {
        return new Tensor((...shape));
    }

    proc _sigmoid(x: real): real {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    proc sigmoid(t: Tensor(?d)): Tensor(d) {
        return t.fmap(_sigmoid);
    }

}




/*

// Have domain of 1d arrays as the index type, that indexes the data
// var dimDom: domain(1,int) = {0..#dim}; // may change
// var idxDom: domain([dimDom] int);
// var data: [idxDom] real;

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