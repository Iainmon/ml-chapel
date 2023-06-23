class C {
    var data: int;
}
class B {
    var y: owned C;
    proc init(in y: C) {
        this.y = y;
    }
}


proc makeB() {
    var x = new owned C(5);
    var z = new B(x);
    return z;
}

var b = makeB();
writeln(b);