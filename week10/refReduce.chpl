record MyInt {
    var x: int;
    proc init(x: int) {
        this.x = x;
    }
}

operator +(a: MyInt, b: MyInt) {
    return new MyInt(a.x + b.x);
}
operator +=(ref a: MyInt, b: MyInt) {
    a.x += b.x;
}

operator +(a: nothing, b: nothing) {
    return none;
}

record Summer {
    var y: (MyInt,MyInt);
    proc sum(ref x: (MyInt,MyInt)) {
        x += y;
    }
}

record ManySummer {
    var summers;
    proc sum(ref x: (MyInt,MyInt)) {
        for param n in 0..#(summers.size) {
            summers[n].sum(x);
        }
    }
}

proc main() {
    var smr = new ManySummer((new Summer((new MyInt(1),new MyInt(2))),new Summer((new MyInt(3),new MyInt(4))), new Summer((new MyInt(5),new MyInt(6)))));
    var x = (new MyInt(0),new MyInt(0));
    forall i in 0..#100 with (+ reduce x) {
        smr.sum(x);
    }
    writeln(x);
    
    x = (new MyInt(0),new MyInt(0));
    for i in 0..#100 {
        smr.sum(x);
    }
    writeln(x);

    var ys = [0..#100] (new MyInt(1),none,none,new MyInt(2));
    var ysum = + reduce ys;
    writeln(ysum);
    
}

