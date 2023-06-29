
// module ChapelArray{
// proc myMethod() {
//     writeln("Hello from ChapelArray");
// }
// }


module Linear {

import LinearAlgebra as LA;
import IO;
import Random;





record Matrix {
    type eltType;
    var matrixDomain: domain(2,int,false);
    var underlyingMatrix: [matrixDomain] eltType; 
    var isVector = false;

    // proc init(A: [?d] ?eltType) where d.rank == 1 {
    //     // this should work
    // }

    proc init(A: [?d] ?eltType) {
        this.eltType = eltType;

        if d.rank == 2 { // A is a matrix
            this.matrixDomain = d;
            this.underlyingMatrix = A;
        } else if d.rank == 1 { // A is a vector
            var mA = LA.transpose(LA.Matrix(A,eltType));
            this.matrixDomain = mA.domain;
            this.underlyingMatrix = mA;
            this.isVector = true;
        } else {
            writeln("Error: Matrix init");
            this.matrixDomain = d;
        }
    }
    proc init(A: [?d] ?t, type eltType) {
        var B: [d] eltType;
        for i in d {
            B[i] = A[i]:eltType;
        }
        var m = new Matrix(B);
        this.eltType = eltType;
        this.matrixDomain = m.matrixDomain;
        this.underlyingMatrix = m.underlyingMatrix;
        this.isVector = m.isVector;
    }

    proc init(type eltType) {
        this.eltType = eltType;
        matrixDomain = {0..1, 0..1};
        underlyingMatrix = LA.Matrix(2,2,eltType);
    }

    // create initializer that accepts an iterator
    // proc init(expr) {
    //     var A = expr;
    //     this.init(A);
    // }


    proc matrix { return underlyingMatrix; }
    
    proc shape { return matrixDomain.shape; }

    iter rows {
        for i in 0..<this.shape[0] do
            yield underlyingMatrix[i,..];
    }

    iter columns {
        for i in 0..<this.shape[1] do
            yield underlyingMatrix[..,i];
    }

    // proc rows


    // proc domain { return matrixDomain; }

    proc writeThis(fw: IO.fileWriter) throws {
        // fw.writeln("matrix(",this.underlyingMatrix,", shape=",this.shape,")");
        var cntr = 0;
        for row in this.rows {
            if cntr == 0 { if this.isVector { fw.write("vector("); } else { fw.write("matrix(");} } else { fw.write("       "); }
            if cntr < this.shape[0] - 1 { fw.writeln(row); } else { fw.write(row); }
            cntr += 1;
        }
        fw.writeln(", shape=",this.shape,")");
    }

    proc transpose() do
        return new Matrix(LA.transpose(underlyingMatrix));
    
    operator +(lhs: Matrix, rhs: Matrix) {
        var A = lhs.matrix + rhs.matrix;
        return new Matrix(A);
    }

    operator +(lhs: Matrix, rhs: eltType) do 
        return new Matrix(lhs.matrix + rhs);

    operator +(lhs: eltType, rhs: Matrix) do
        return new Matrix(lhs + rhs.matrix);

    operator +=(ref lhs: Matrix, const ref rhs: Matrix) {
        lhs.underlyingMatrix += rhs.underlyingMatrix;
    }

    operator -(lhs: Matrix, rhs: Matrix) {
        var A = lhs.matrix - rhs.matrix;
        return new Matrix(A);
    }

    operator *(lhs: Matrix, rhs: Matrix) {
        var A = lhs.matrix * rhs.matrix;
        return new Matrix(A);
    } 
    operator *(r: real, m: Matrix(real)) {
        var A = r * m.matrix;
        // writeln((m.matrix * r).type:string);
        return new Matrix(A);
    }

    operator *=(ref lhs: Matrix, const ref rhs: Matrix) {
        lhs.matrix *= rhs.matrix;
    }

    operator *(lhs: Matrix, rhs: eltType) do
        return new Matrix(lhs.matrix * rhs);
    
    operator *(lhs: eltType, rhs: Matrix) do
        return new Matrix(lhs * rhs.matrix);

    proc dot(rhs: Matrix) {
        var (m1,n1) = this.shape;
        var (m2,n2) = rhs.shape;
        if n1 != m2 {
            writeln("Cannot multiply: ", this, " and ", rhs);
            assert(n1 == m2, "Error computing dot product.");
        }
        return new Matrix(LA.dot(this.matrix,rhs.matrix));
    }

    proc dot(rhs: [?d] eltType) {
        var B = new Matrix(rhs);
        return new Matrix(LA.dot(this.matrix,B.matrix));
    }

    proc vector {
        if this.isVector {
            var (m,n) = this.shape;
            var vec: [0..#m] eltType;
            for i in 0..<m {
                vec[i] = this.underlyingMatrix[i,0];
            }
            return vec;
        } else {
            halt("Error: matrix is not a vector.");
        }
    }
}

proc matrixFromRows(arrays ...?n, type eltType) {
    var d = {0..#(arrays.size), 0..#(arrays[0].domain.size)};
    var A = LA.Matrix(d,eltType);
    foreach (i,j) in d do
        A[i,j] = arrays[i][j] : eltType;
    return new Matrix(A);
}

proc matrixFromColumns(arrays ...?n, type eltType) {
    var d = {0..#(arrays[0].domain.size), 0..#(arrays.size)};
    var A = LA.Matrix(d,eltType);
    foreach (i,j) in d do
        A[i,j] = arrays[j][i] : eltType;
    return new Matrix(A);
}

proc vectorToMatrix(vector: [?d] ?t, type eltType=real) where d.rank == 1 {
    return new Matrix(vector,eltType);
}


proc zeros(m: int, n: int, type eltType=real) {
    var A = LA.Matrix(m,n,eltType);
    return new Matrix(A);
}

proc zeroVector(n: int, type eltType=real) {
    var A = LA.Vector(n,eltType);
    return new Matrix(A);
}

proc random(m: int, n: int, type eltType=real) {
    var A = LA.Matrix(m,n,eltType);
    var rng = new owned Random.RandomStream(eltType=eltType);
    rng.fillRandom(A);
    return new Matrix(A);
}

proc randomVector(n: int, type eltType=real) {
    var A = LA.Vector(n,eltType);
    var rng = new owned Random.RandomStream(eltType=eltType);
    rng.fillRandom(A);
    return new Matrix(A);
}
















proc main() {

    var m1 = random(10,10);
    var _x = 1.0 * m1;
    writeln(_x);
    /*
    writeln(LA.Matrix([1,3]));


    var m: Matrix(real);
    m = new Matrix(real);

    var marr: [0..#5] Matrix(real);
    writeln(marr[0]);

    var zz = zeros(2,3);
    writeln(zz);

    // var v = LA.transpose(LA.Matrix(LA.Vector([1,0,0],real)));
    // // or // var v = LA.Vector([1,0,0],real);
    // writeln(v);
    // writeln(v.shape);
    // var A = LA.Matrix([1,2,3],[4,5,6],real);
    // writeln(A);
    // writeln(A.shape);
    // var x = LA.dot(A,v);
    // writeln(x);


    var A2 = matrixFromRows([1,2,3],[4,5,6],[7,8,9],real); // Create a matrix with listed rows
    writeln(A2);

    var v2 = vectorToMatrix([1,0,0]);// new Matrix([1,0,0],real); // Create a vector
    writeln(v2);

    var x2 = A2.dot(v2); // Matrix-vector multiplication
    writeln(x2);

    var size = 50;

    var m2 = random(5000,5000);
    var z = m1.dot(m2);

    foreach i in 1..10000 {
        z = m1.dot(z);
        writeln(i, " ", z.shape);
    }*/
}

}
