// scalar variable
var number: real = 3.14;

// array with implicit domain
var a = [0, 1, 2, 3];
writeln(a.domain);

// array with explicit domain
var a1: [0..<10] int;
writeln(a1);

// explicit domain as a variable
const d = {0..<x, 1..10 by 2};
writeln("domains type: ", d.type:string);
writeln(d);
var a2: [d] real;
var a3: [d] int;

// loop over domain
forall (i, j) in d {
    a2[i, j] = a3[i, j] * 2;
}

// self referential class
class C {
    var x: int;
    var other: owned C? = nil;
}

// owned and shared classes
var oc = new owned C(1);
var sc = new shared C(1);
