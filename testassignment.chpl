

record Mat {
    type eltType;
    var dom: domain(2,int);
    var underlying: [dom] eltType;
    proc init(type eltType) {
        this.eltType = eltType;
        dom = [0..#0, 0..#0];
        underlying = 0;
    }
    proc init(M: [?d] ?t) {
        eltType = t;
        dom = d;
        underlying = M;
    }

    operator +=(ref lhs: Mat, rhs: Mat) {
        lhs.underlying += rhs.underlying;
    }

}


proc main() {
    var m: [0..#3, 0..#3] int = 1;
    var A = new Mat(m);
    var B = new Mat(m);
    A += B;
    writeln(A);

    var As = [i in 0..5] new Mat(m);
    var Bs = [i in 0..5] new Mat(m);
    As += Bs;
    writeln(As);
}