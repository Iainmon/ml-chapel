
use Random;
import LinearAlgebra as LA;
// use Math;
use List;
import Linear as lina;
import Math;


proc list.get(i: int) ref { return this[mod(i,this.size)]; }
proc list.getIdx(i: int) { return mod(i,this.size); }
proc getIdx(xs: [?d] ?t, i: int) {
    return mod(i,xs.domain.dim(0).size);
}


/* Sigmoid functions */

proc sigmoid(x: real(64)): real(64) {
    return 1.0 / (1.0 + Math.exp(-x));
    // return (Math.exp(x) / (Math.exp(x) + 1.0)) + 0.001;
}
proc sigmoid(m: lina.Matrix(real(64))): lina.Matrix(real(64)) {
    const A = sigmoid(m.matrix);
    return new lina.Matrix(A);
}
proc sigmoid(m: lina.Vector(real(64))): lina.Vector(real(64)) {
    const A = sigmoid(m.vector);
    return new lina.Vector(A);
}

proc sigmoidM(m: lina.Matrix(real(64))): lina.Matrix(real(64)) {
    const A = sigmoid(m.matrix);
    return new lina.Matrix(A);
}

proc sigmoidPrime(x: real(64)): real(64) {
    const sig = sigmoid(x);
    return sig * (1.0 - sig);

    // return sig * (1.001 - sig);
}

proc sigmoidPrimeM(m: lina.Matrix(real(64))): lina.Matrix(real(64)) {
    const A = sigmoidPrime(m.matrix);
    return new lina.Matrix(A);
}

proc sigmoidPrimeM(in V: lina.Vector(real(64))): lina.Vector(real(64)) {
    const A = sigmoidPrime(V.underlyingVector);
    // V.vectorDomain = A.domain;
    // V.underlyingVector = A;
    // return V;
    return new lina.Vector(A);
}
proc sigmoidPrime(const ref m: lina.Matrix(real(64))): lina.Matrix(real(64)) {
    const A = sigmoidPrime(m.underlyingMatrix);
    return new lina.Matrix(A);
}
proc sigmoidPrime(const ref V: lina.Vector(real(64))): lina.Vector(real(64)) {
    const A = sigmoidPrime(V.underlyingVector);
    return new lina.Vector(A);
}

proc err(args...?n) {
    var s = "";
    for param i in 0..<n {
        s += args(i): string;
    }
    try! throw new Error(s);
}

proc main() {
    for i in 0..80000 {
        var ir = (i: real(64)) / 1000.0;
        var y = sigmoid(ir);
        writeln("(", i, ",", y ,")");
        if abs(y - 1) < 0.000000000000000000001 {
            writeln("Stopped at ", i);
            writef("%.15r, %.15r", i, y);
            break;
        }
    }
}