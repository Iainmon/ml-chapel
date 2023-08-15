module Tensor {

    import Linear as la;
    import Math;
    import IO;
    import IO.FormattedIO;
    import ChapelArray;

    param debugPrint = false;

    proc err(args...?n) {
        var s = "";
        for param i in 0..<n {
            s += args(i): string;
        }
        try! throw new Error(s);
    }
    proc debugWrite(args...?n) {
        if debugPrint {
            var s = "";
            for param i in 0..<n {
                s += args(i): string;
            }
            try! IO.stdout.write(s);
            try! IO.stdout.flush();
        }
    }

    iter cartesian(X,Y) {
        for x in X {
            for y in Y {
                yield (x,y);
            }
        }
    }
    iter cartesian(param tag: iterKind,X,Y) where tag == iterKind.standalone {
        forall x in X {
            forall y in Y {
                yield (x,y);
            }
        }
    }



    proc domainFromShape(shape: int ...?d): domain(d,int) {
        var ranges: d*range;
        for (size,r) in zip(shape,ranges) do r = 0..#size;
        return {(...ranges)};
    }

    proc nbase(bounds: ?rank*int, n: int): rank*int {
        var filled: rank*int;
        var idx: int = rank - 1;
        var curr: int = 0;
        var carry: bool = false;
        while curr < n {
            filled[idx] += 1;
            if filled[idx] >= bounds[idx] {
                carry = true;
                filled[idx] = 0;
                idx -= 1;
                if idx < 0 then err("Error in nbase: ", n," is too large for bounds.");
            } else {
                carry = false;
                idx = rank - 1;
                curr += 1;
            }
        }
        return filled;
    }
    
    record Tensor {
        param rank: int;
        type eltType = real(64);

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
        // proc init(type eltType, shape: int ...?dim) {
        //     this.rank = dim ;
        //     this.eltType = eltType;
        //     var ranges: dim*range;
        //     for (size,r) in zip(shape,ranges) do r = 0..#size;
        //     this._domain = {(...ranges)};
        // }
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

        proc init(itr) where itr.type:string == "promoted expression" || itr.type:string == "iterator" {
            const A = itr;
            this.init(A);
            writeln("init(iter)");
        }


        proc init=(other: Tensor(?rank,?eltType)) {
            this.rank = other.rank;
            this.eltType = other.eltType;
            this._domain = other._domain;
            this.data = other.data;
            // writeln("init= called");
        }

        operator =(ref lhs: Tensor(?rank,?eltType), rhs: Tensor(rank,eltType)) {
            lhs._domain = rhs._domain;
            lhs.data = rhs.data;
        }


        operator =(ref lhs: Tensor(?rank,?eltType), rhs: [?d] eltType) where d.rank == rank {
            lhs.reshapeDomain(d);
            lhs.data = rhs;
        }
        proc init=(rhs: [?d] eltType) where d.rank == rank {
            this.init(d.rank,eltType);
            this.reshapeDomain(d);
            this.data = rhs;
        }
        operator :(from: [?d] ?eltType, type toType: Tensor(d.rank,eltType)) {
            var t: Tensor(d.rank,eltType) = from;
            return t;
        }

        // operator =(ref lhs: Tensor(?rank,?eltType), in rhs: ?it) where (isRefIterType(it) || (isArray(rhs) && rhs.eltType == eltType)) && rhs.rank == rank {
        //     lhs.reshapeDomain(rhs.domain);
        //     lhs.data = rhs;
        // }
        // proc init=(in rhs: ?it) where isRefIterType(it) || isArray(rhs) {
        //     this.init(rank,eltType);
        //     this.reshapeDomain(rhs.domain);
        //     this.data = rhs;
        // }
        // operator :(in from: ?it, type toType: Tensor(?rank,?eltType)) where (isRefIterType(it) || (isArray(from) && from.eltType == eltType)) && from.rank == rank {
        //     // compilerError("Cannot convert from ",from.type:string," to ",toType:string);
        //     var t: Tensor(rank,eltType) = from;
        //     return t;
        // }

        operator =(ref lhs: Tensor(?rank,?eltType), itr) where itr.type:string == "promoted expression" || itr.type:string == "iterator" {
            lhs.reshapeDomain(itr.domain);
            lhs.data = itr;
        }
        proc init=(itr) where itr.type:string == "promoted expression" || itr.type:string == "iterator" {
            const A = itr;
            this.init(A);
        }
        operator :(itr, type toType: Tensor(?rank,?eltType)) where itr.type:string == "promoted expression" || itr.type:string == "iterator" {
            var t: Tensor(rank,eltType) = itr;
            return t;
        }




        forwarding data only this;
        // forwarding data only domain;

        proc reshapeDomain(d: this._domain.type) do
            this._domain = d;
        

        proc transpose() where rank == 1 {
            const (p,) = shape;
            var t = new Tensor(eltType,1,p);
            t.data[0,..] = this.data;
            return t;
        }
        proc transpose() where rank == 2 {
            const (m,n) = this.shape;
            // var M: [0..#n,0..#m] eltType; 
            // forall (i,j) in M.domain {
            //     M[i,j] = this.data[j,i];
            // }
            // return new Tensor(M);
            var M = new Tensor(2,eltType);
            M._domain = {0..#n,0..#m};
            forall (i,j) in M.data.domain with (ref M, ref this) {
                M[i,j] = this[j,i];
            }
            return M;
        }



        proc normalize() {
            const norm = sqrt(frobeniusNormPowTwo(this));
            const data = this.data / norm;
            return new Tensor(data);

        }

        proc flatten() {
            const size = this.data.domain.size;
            const flatD = {0..#size};
            // const v = ChapelArray.reshape(this.data,flatD);
            const v = for (i,a) in zip(flatD,this.data) do a;
            return new Tensor(v);
        }

        proc reshape(shape: int ...?d) {
            const dom = domainFromShape((...shape));
            // const data = ChapelArray.reshape(this.data,dom);
            const data = for (i,a) in zip(dom,this.data) do a;
            return new Tensor(data);
        }

        
        proc fmap(fn) {
            const data = fn(this.data);
            return new Tensor(data);
        }
        
        proc writeThis(fw: IO.fileWriter) throws {
            fw.write("tensor(");
            const shape = this.shape;
            var first: bool = true;
            for (x,i) in zip(data,0..) {
                const idx = nbase(shape,i);
                if idx[rank - 1] == 0 {
                    if !first {
                        fw.write("\n       ");
                    }
                    fw.write("[");
                }
                fw.writef("%{##.##########}",x);
                
                if idx[rank - 1] < shape[rank - 1] - 1 {
                    if rank == 1 then
                        fw.write("\n        ");
                    else
                        fw.write("  ");
                } else {
                    fw.write("]");
                }

                // if rank == 2 {
                //     if idx[rank - 2] < shape[rank - 2] - 1 && i < data.domain.size - 1 then
                //         fw.writeln(",");
                // }
                // if rank == 3 {
                //     if i % (shape[1] * shape[2]) + 1 == 0 {
                //         fw.writeln(",");
                //     }
                // }
                first = false;
            }
            fw.writeln(", shape=",this.shape,")");
        }

        proc write(fw: IO.fileWriter) throws {
            fw.write(rank);
            for s in shape do
                fw.write(s:int);
            for i in data.domain do
                fw.write(data[i]);
        }
        proc read(fr: IO.fileReader) throws {
            var r = fr.read(int);
            if r != rank then
                err("Error reading tensor: rank mismatch.", r , " != this." , rank);
            var s = this.shape;
            for i in 0..#rank do
                s[i] = fr.read(int);
            var d = domainFromShape((...s));
            this._domain = d;
            for i in d do
                this.data[i] = fr.read(eltType);
                
        }
    }

    // operator =(ref lhs: Tensor(?rank,?eltType), in rhs) {
    //     lhs._domain = rhs._domain;
    //     lhs.data = rhs;
    // }

    // operator =(ref lhs: Tensor(?rank,?eltType), rhs: [?d] eltType) where d.rank == rank {
    //     lhs.reshapeDomain(d);
    //     lhs.data = rhs.data;
    // }

    // operator =(ref lhs: Tensor(?rank,?eltType), rhs: Tensor(rank,eltType)) {
    //     lhs._domain = rhs._domain;
    //     lhs.data = rhs.data;
    // }

    // operator =(ref lhs: Tensor(?rank,?eltType), rhs: Tensor(rank,?t)) {
    //     lhs._domain = rhs._domain;
    //     lhs.data = (rhs.data : eltType);
    // }
    

    operator +(lhs: Tensor(?rank,?eltType), rhs: Tensor(rank,eltType)) {
        // Fastest
        var t = new Tensor(rank=rank,eltType=eltType);
        t.reshapeDomain(lhs._domain);
        t.data = lhs.data + rhs.data;
        return t; // 3.2, 3.2, 4.0

        // return new Tensor(lhs.data + rhs.data); // 4.5, 4.0

        // var t = new Tensor(rank=d,eltType=ty);
        // t.reshapeDomain(lhs._domain);
        // t = lhs.data + rhs.data;
        // return t; // 4.6, 5.0

        // var t = new Tensor(rank=d,eltType=ty);
        // t.reshapeDomain(lhs._domain);
        // const data = lhs.data + rhs.data;
        // t.data = data;
        // return t; // 4.9, 4.6

        // var t = new Tensor(rank=d,eltType=ty);
        // const data = lhs.data + rhs.data;
        // t = data;
        // return t; // 4.6,4.7

        // var t = new Tensor(rank=d,eltType=ty);
        // t = lhs.data + rhs.data;
        // return t; // 4.5,4.6
    }
    operator +=(ref lhs: Tensor(?d), const ref rhs: Tensor(d)) {
        lhs.data += rhs.data;
    }
    operator +=(ref lhs: Tensor(?rank,?eltType), rhs) where (isArray(rhs) && rhs.rank == rank) || rhs.type == eltType {
        lhs.data += rhs;
    }
    operator +=(ref lhs: Tensor(?rank,?eltType), rhs) where rhs.type:string == "promoted expression" || rhs.type:string == "iterator" {
        lhs.data += rhs;
    }
    operator -(lhs: Tensor(?rank,?eltType), rhs: Tensor(rank,eltType)) {
        var t = new Tensor(rank=rank,eltType=eltType);
        t.reshapeDomain(lhs._domain);
        t.data = lhs.data - rhs.data;
        return t;
    }
    operator -=(ref lhs: Tensor(?d), const ref rhs: Tensor(d)) {
        lhs.data -= rhs.data;
    }
    operator -=(ref lhs: Tensor(?rank,?eltType), rhs) where (isArray(rhs) && rhs.rank == rank) || rhs.type == eltType {
        lhs.data -= rhs;
    }
    operator *(c: ?eltType, rhs: Tensor(?rank,eltType)) {
        var t = new Tensor(rank=rank,eltType=eltType);
        t.reshapeDomain(rhs._domain);
        t.data = c * rhs.data;
        return t;
    }
    operator *(lhs: Tensor(?rank,?eltType), c: eltType) {
        var t = new Tensor(rank=rank,eltType=eltType);
        t.reshapeDomain(lhs._domain);
        t.data = lhs.data * c;
        return t;
    }
    operator *(lhs: Tensor(?rank,?eltType), rhs: Tensor(rank,eltType)) {
        // Hermitian product, not composition
        var t = new Tensor(rank=rank,eltType=eltType);
        t.reshapeDomain(lhs._domain);
        t.data = lhs.data * rhs.data;
        return t;
    }
    operator *=(ref lhs: Tensor(?d), const ref rhs: Tensor(d)) {
        lhs.data *= rhs.data;
    }
    operator *=(ref lhs: Tensor(?rank,?eltType), rhs) where (isArray(rhs) && rhs.rank == rank) || rhs.type == eltType {
        lhs.data *= rhs;
    }

    operator *(lhs: Tensor(2,?eltType), rhs: Tensor(1,eltType)): Tensor(1,eltType) {
        const (m,n) = lhs.shape;
        const (p,) = rhs.shape;
        if n != p then
            err("Trying to apply a matrix of shape ",lhs.shape, " to a vector of shape ", rhs.shape);
        
        // const a = lhs.data;
        // const v = rhs.data;
        // var w: [0..#m] eltType;
        // forall i in 0..#m {
        //     const row = a[i,..];
        //     w[i] = + reduce (row * v);
        // }
        // return new Tensor(w); // .3, .5, .3


        const a = lhs.data;
        const v = rhs.data;
        var w = new Tensor(rank=1,eltType=eltType);
        w.reshapeDomain({0..#m});
        forall i in 0..#m with (ref w) {
            const row = a[i,..];
            w[i] = + reduce (row * v);
        }
        return w;

        // var w = new Tensor(rank=1,eltType=eltType);
        // w.reshapeDomain({0..#m});
        // forall i in 0..#n with (ref w) {
        //     var ci = 0.0;
        //     forall k in 0..#m with (+ reduce ci) {
        //         ci += lhs[k,i] * rhs[k];
        //     }
        //     w[i] = ci;
        // } // .4, .3, .4

        // var w = new Tensor(rank=1,eltType=eltType);
        // w.reshapeDomain({0..#m});
        // forall i in 0..#n with (+ reduce w) {
        //     w += lhs[..,i] * rhs.data;
        // } // .2, .3, .3

        // const v = rhs.data;
        // foreach i in 0..#m {
        //     w[i] = + reduce (lhs[i,..] * v);
        // } // .8, .7, .8

        //return w;
    }

    operator *(lhs: Tensor(1,?eltType), rhs: Tensor(2,eltType)): Tensor(2,eltType) {
        const (p,) = lhs.shape;
        const (m,n) = rhs.shape;
        if m != 1 then
            err("Trying to apply a vector of shape ",lhs.shape, " to a matrix of shape ", rhs.shape);
        
        // const a = rhs.data;
        // const v = lhs.data;
        // var b: [0..#p, 0..#n] eltType;
        // forall (i,j) in b.domain {
        //     b[i,j] = v[i] * a[0,j];
        // }
        // return new Tensor(b); // .4, .4, .2

        var b = new Tensor(rank=2,eltType=eltType);
        b.reshapeDomain({0..#p, 0..#n});
        foreach (i,j) in {0..#p, 0..#n} {
            b[i,j] = lhs[i] * rhs[0,j];
        }
        return b; 
    }

    operator /(lhs: Tensor(?d,?eltType), c: eltType) {
        const data = lhs.data / c;
        return new Tensor(data);
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
    proc _sigmoidPrime(x: real): real {
        const s = _sigmoid(x);
        return s * (1.0 - s);
    }

    proc sigmoid(t: Tensor(?d)): Tensor(d) {
        return t.fmap(_sigmoid);
    }
    proc sigmoidPrime(t: Tensor(?d)): Tensor(d) {
        return t.fmap(_sigmoidPrime);
    }
    proc frobeniusNormPowTwo(t: Tensor(?d)): real {
        const AA = t.data ** 2.0;
        return + reduce AA;
    }

    proc exp(t: Tensor(?d)): Tensor(d) {
        var data: [t.data.domain] real;
        forall i in data.domain do
            data[i] = Math.exp(t.data[i]);
        return new Tensor(data);
        // var t = new Tensor(rank=d,eltType=real);
        // t.data = [x in t.data] Math.exp(x);
        // return t;
    }

    proc argmax(A: [?d] real) where d.rank == 1 {
        // var max: real = A[0];
        var am: int = 0;
        for i in A.domain {
            if A[i] > A[am] {
                // max = A[i];
                am = i;
            }
        }
        return am;
    }

    proc convolve(kernel: [?dk] ?eltType, X: [?dx] eltType) where dx.rank == 2 && dk.rank == 2 {
        const (h,w) = X.shape;
        const (kh,kw) = kernel.shape;
        const newH = h - (kh - 1);
        const newW = w - (kw - 1);
        var Y: [0..#newH,0..#newW] eltType;
        // forall (i,j) in Y.domain with (var region: [0..#kh,0..#kw] eltType) {
        //     region = X[i..#kh, j..#kw];
        //     Y[i,j] = + reduce (region * kernel);
        // }

        forall (i,j) in Y.domain {
            var sum = 0.0;
            forall (k,l) in kernel.domain with (+ reduce sum) {
                sum += X[i + k, j + l] * kernel[k,l];
            }
            Y[i,j] = sum;
        }
        return Y;
    }


    proc convolveRotateRefPadding(const ref kernel: [?dk] ?eltType, const ref X: [?dx] eltType, ref Y: [?dy] eltType) where dx.rank == 2 && dk.rank == 2 {
        const (h,w) = X.shape;
        const (kh,kw) = kernel.shape;
        const newH = h - (kh - 1);
        const newW = w - (kw - 1);
        // var Y: [0..#newH,0..#newW] eltType;
        
        forall (i,j) in Y.domain {
            var sum = 0.0;
            forall (k,l) in kernel.domain with (+ reduce sum) {
                sum += X[h - i - k - 1, h - j - l - 1] * kernel[k,l];
            }
            Y[i,j] = sum;
        }
        // return Y;
    }

    proc convolveRotate(kernel: [?dk] ?eltType, X: [?dx] eltType) where dx.rank == 2 && dk.rank == 2 {
        const (h,w) = X.shape;
        const (kh,kw) = kernel.shape;
        const newH = h - (kh - 1);
        const newW = w - (kw - 1);
        var Y: [0..#newH,0..#newW] eltType;
        
        forall (i,j) in Y.domain {
            var sum = 0.0;
            forall (k,l) in kernel.domain with (+ reduce sum) {
                sum += X[i + k, j + l] * kernel[kh - k - 1, kw - l - 1];
            }
            Y[i,j] = sum;
        }
        return Y;
    }

    proc convolve(kernel: Tensor(2), X: Tensor(2)): Tensor(2) {
        return new Tensor(convolve(kernel.data,X.data));
    }

    proc rotate180(kernel: [?d] ?eltType) where d.rank == 2 {
        const (kh,kw) = kernel.shape;
        var ker: [0..#kh,0..#kw] eltType;
        forall (i,j) in ker.domain {
            ker[i,j] = kernel[kh - i - 1, kw - j - 1];
        }
        return ker;
    }

    proc rotate180(kernel: Tensor(2)): Tensor(2) {
        return new Tensor(rotate180(kernel.data));
    }

    proc fullConvolve(kernel: [?dk] ?eltType, X: [?dx] eltType) where dx.rank == 2 && dk.rank == 2 {
        const (h,w) = X.shape;
        const (kh,kw) = kernel.shape;
        const (paddingH,paddingW) = (kh - 1,kw - 1);
        const newH = h + 2 * paddingH;
        const newW = w + 2 * paddingW;
        var Y: [0..#newH,0..#newW] eltType;
        Y = 0.0;
        Y[paddingH..#h, paddingW..#w] = X;
        return convolve(kernel,Y);
    }

    proc fullConvolve(kernel: Tensor(2), X: Tensor(2)): Tensor(2) {
        return new Tensor(fullConvolve(kernel.data,X.data));
    }


    proc pad(const ref x: Tensor(2), padding: int) {
        var t = new Tensor(2,real);
        const (h,w) = x.shape;
        t.reshapeDomain({0..#(h + 2 * padding),0..#(w + 2 * padding)});
        t.data[padding..#h, padding..#w] = x.data;
        return t;
    }
    proc pad(const ref x: Tensor(3), padding: int) {
        var t = new Tensor(3,real);
        const (h,w,c) = x.shape;
        t.reshapeDomain({0..#(h + 2 * padding),0..#(w + 2 * padding),0..#c});
        forall (i,j,c) in x.data.domain with (ref t) {
            t[i + padding,j + padding,c] = x[i,j,c];
        }
        return t;
    }
    proc correlate(const ref filter: Tensor(?), const ref input: Tensor(?), stride: int = 1, padding: int = 0) {
        if padding > 0 then 
            return correlate_(filter,pad(input,padding),stride,padding);
        return correlate_(filter,input,stride,padding);
    }
    proc correlate_(const ref filter: Tensor(2), const ref input: Tensor(2), stride: int, padding: int): Tensor(2) {
        const (kh,kw) = filter.shape;
        const (nh,nw) = input.shape;
        if kh != kw then err("Correlation only works with square filters.", kh, " != ", kw);
        const (outH,outW): 2*int = ((nh - kh + padding + stride) / stride,(nw - kw + padding + stride) / stride);

        var corr = new Tensor(2,real);
        corr.reshapeDomain({0..#outH,0..#outW});

        forall (x,y) in corr.data.domain with (ref corr) {
            var sum = 0.0;
            forall (i,j) in filter.data.domain with (+ reduce sum) {
                sum += input[x * stride + i, y * stride + j] * filter[i,j];
            }
            corr[x,y] = sum;
        }

        return corr;
    }

    proc correlate_(const ref filter: Tensor(3), const ref input: Tensor(3), stride: int, padding: int): Tensor(2) {
        const (kh,kw,cIn) = filter.shape;
        const (nh,nw,nc) = input.shape;
        if kh != kw then err("Correlation only works with square filters.", kh, " != ", kw);
        if cIn != nc then err("Correlation only works with filters and inputs of the same depth.", cIn, " != ", nc);

        const (outH,outW): 2*int = ((nh - kh + padding + stride) / stride,(nw - kw + padding + stride) / stride);

        var corr = new Tensor(2,real);
        corr.reshapeDomain({0..#outH,0..#outW});

        forall (x,y) in corr.data.domain with (ref corr) {
            var sum = 0.0;
            forall (i,j,c) in filter.data.domain with (+ reduce sum) {
                sum += input[x * stride + i, y * stride + j,c] * filter[i,j,c];
            }
            corr[x,y] = sum;
        }

        return corr;
    }
    // proc correlateDelta(pIn: 2*int,pOut: 2*int,const ref filter: Tensor(2), inputDomain: domain(2,int), stride: int = 1, padding: int = 0) {
    //     const (m,n) = pIn;
    //     const (x,y) = pOut;
    //     if filter.domain.contains((m - i, n - j))
    // }


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