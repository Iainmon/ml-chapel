// class C {
//     var i: int;
//     proc init() { this.i = 0; }
//     proc init(i: int) { this.i = i; }
//     operator +(a: C, b: C) { return new C(a.i + b.i); }
// }

// var is = [1,2,3,4,5];
// var cs = [i in is] (new C(i));
// var sum = + reduce cs;
// writeln(sum);

class C {
    var i: int;
    proc init() { this.i = 0; }
    proc init(i: int) { this.i = i; }
    operator +(a: C, b: C) { return new C(a.i + b.i); }
    operator +(a: C?, b: C?) {
        // writeln(a,b);
        if a == nil && b == nil then
            return new C(): C?;

        if a == nil then
            return new C(b!.i): C?;
        if b == nil then
            return new C(a!.i): C?;

        return new C(a!.i + b!.i): C?;
    }
}

var is = [1,2,3,4,5];
var cs: [is.domain] owned C? = [i in is] (new C(i));
var sum = + reduce cs;
writeln(sum);