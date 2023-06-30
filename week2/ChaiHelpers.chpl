
use Random;
import LinearAlgebra as LA;
use Math;
use List;
import Linear as lina;

proc list.get(i: int) ref { return this[mod(i,this.size)]; }
proc list.getIdx(i: int) { return mod(i,this.size); }
proc getIdx(xs: [?d] ?t, i: int) {
    return mod(i,xs.domain.dim(0).size);
}


/* Sigmoid functions */

proc sigmoid(x: real(64)): real(64) do
    return 1.0 / (1.0 + exp(-x));

proc sigmoidM(m: lina.Matrix(real(64))): lina.Matrix(real(64)) {
    var A = sigmoid(m.matrix);
    return new lina.Matrix(A);
}

proc sigmoidPrime(x: real(64)): real(64) {
    var sig = sigmoid(x);
    return sig * (1.0 - sig);
}

proc sigmoidPrimeM(m: lina.Matrix(real(64))): lina.Matrix(real(64)) {
    var A = sigmoidPrime(m.matrix);
    return new lina.Matrix(A);
}




