

record Summer {
    var y: int;
    proc sum(ref x: int) {
        x += y;
    }
}

record ManySummer {
    var summers;
    proc sum(ref x: int) {
        for param n in 0..#(summers.size) {
            summers[n].sum(x);
        }
    }
}

proc main() {
    var smr = new ManySummer((new Summer(1),new Summer(2), new Summer(3)));
    var x = 0;
    forall i in 0..#100 with (+ reduce x) {
        smr.sum(x);
    }
    writeln(x);
    
    x = 0;
    for i in 0..#100 {
        smr.sum(x);
    }
    writeln(x);
}

