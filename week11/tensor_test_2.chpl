

// import Tensor as tn;
// use Tensor only Tensor;



// proc foo(x: Tensor(2)) {
//     var t = x;
//     t.data += 1;
//     return t;
// }

// var a = tn.zeros(3,3,3);
// var b = tn.zeros(3,3);

// for i in 0..#3 {
//     const a_i = new Tensor(a[..,..,i]);
//     const c = foo(a_i);
//     b.data += c.data;
// }

// writeln(b);


proc f(t: [?dIn] real): [domain(1,int)] real where dIn.rank == 1 {
    return t;
}

proc g(t: [domain(1,int)] real): [domain(1,int)] real {
    return t;
}

var x: [0..10] real = 0..10;
writeln(f(x));
writeln(g(x));