

proc _array.hello() {
    writeln("hello");
}

// type tensor = [] real;
proc tensor(d,type eltType) type {
    return [d] eltType;
}

type Tensor = tensor(?,real);

proc foo(t: Tensor(?d)) {
    writeln(d);
}
proc main() {
    var a: [0..10] int;
    a.hello();
}