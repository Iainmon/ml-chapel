



// Micrograd port to Chapel

/*
a = Value()
b = Value()
c = f(a,b)
c.backward()
a.grad = dc/da
b.grad = dc/db

*/

use Map;
use List;
use Set;
use Math;

class Expr {
    proc freeVars(): set(string) { return new set(string); }
    proc grad(name: string): real { return 0; }
    proc value(): real { return 0; }

    proc init() {
        
    }

    proc showGradient(myName: string) {
        for name in this.freeVars() {
            var grad = this.grad(name);
            writeln("d", myName,"/d", name, " = ", grad);
        }
        writeln("------------------------");
    }

    proc nudge(d: map(string,real)) { }

    operator +(in a: Expr, in b: Expr) { return (new AddExpr(a,b)): owned Expr; }
    operator -(in a: Expr, in b: Expr) { return (new SubExpr(a,b)): owned Expr; }
    operator *(in a: Expr, in b: Expr) { return (new MulExpr(a,b)): owned Expr; }
    operator /(in a: Expr, in b: Expr) { return (new DivExpr(a,b)): owned Expr; }
    operator **(in a: Expr, c: real) { return (new ExpExpr(a,c)): owned Expr; }
    proc relu() { return (new owned ReluExpr(this)): Expr; }
}

class Literal: Expr {
    var name: string;
    var base: real;

    proc init() {
        this.name = "";
        this.base = 0;
    }
    proc init(name: string, base: real) {
        this.name = name;
        this.base = base;
    }

    override proc freeVars() do
        return new set(string,[this.name]);

    override proc value() do
        return this.base;

    override proc grad(name: string): real {
        if this.name == name then
            return 1;
        return 0;
    }
    override proc nudge(d: map(string,real)) {
        var diff = d[this.name];
        var learnRate = 0.3;
        this.base +=  -diff * learnRate;
    }
}

proc lit(name: string, value: real): Expr {
    return (new owned Literal(name,value)): owned Expr;
}

class Constant: Expr {
    var base: real;
    override proc value() { return this.base; }
    proc init() { this.base = 0;}
    proc init(base: real) { this.base = base; }
}


class Compound: Expr {
    var left: owned Expr;
    var right: owned Expr;

    proc init() {
        this.left = new Expr();
        this.right = new Expr();
    }

    proc binOpInit(in a: Expr, in b: Expr) {
        this.left = a;
        this.right = b;
    }

    override proc freeVars() {
        return this.left.freeVars() + this.right.freeVars();
    }

    override proc nudge(d: map(string,real)) {
        this.left.nudge(d);
        this.right.nudge(d);
    }
}

class AddExpr: Compound {
    proc init(in a: Expr, in b: Expr) {
        this.binOpInit(a,b);
    }
    override proc value() { return this.left.value() + this.right.value(); }
    override proc grad(name: string) {
        return this.left.grad(name) + this.right.grad(name);
    }
}
class SubExpr: Compound {
    proc init(in a: Expr, in b: Expr) { this.binOpInit(a,b); }
    override proc value() { return this.left.value() - this.right.value(); }
    override proc grad(name: string) {
        return this.left.grad(name) - this.right.grad(name);
    }
}
class MulExpr: Compound {
    proc init(in a: Expr, in b: Expr) { this.binOpInit(a,b); }
    override proc value() { return this.left.value() * this.right.value(); }
    override proc grad(name: string) {
        return this.left.grad(name) * this.right.value() + this.right.grad(name) * this.left.value();
    }
}
class DivExpr: Compound {
    proc init(in a: Expr, in b: Expr) { this.binOpInit(a,b); }
    override proc value() { return this.left.value() / this.right.value(); }
    override proc grad(name: string) {
        var f = this.left.borrow();
        var fx = f.value();
        var df = f.grad(name);

        var g = this.right.borrow();
        var gx = g.value();
        var dg = g.grad(name);

        return (df * gx - dg * fx) / (gx ** 2);
    }
}


class ExpExpr: Compound {
    proc init(in a: Expr, c: real) {
        var cexpr = cnst(c);
        this.binOpInit(a,cexpr);
    }
    override proc value() { return this.left.value() ** this.right.value(); }
    override proc grad(name: string) {
        var c = this.right.value();
        return c * (this.left.value() ** (c - 1)) * this.left.grad(name);
    }
}

class ReluExpr: Expr {
    var x: owned Expr;
    override proc freeVars() {
        return this.x.freeVars();
    }
    override proc value() {
        // if this.x.value() < 0 { return 0; }
        // return this.x.value();
        return tanh(this.x.value());
    }
    override proc grad(name: string) {
        // if this.x.value() < 0 { return 0; }
        // return this.x.value() * this.x.grad(name);
        var fx = this.x.value();
        var df = this.x.grad(name);
        return (1 / (cosh(fx) ** 2)) * df;
    }
}

proc relu(in e: Expr) {
    return new owned ReluExpr(e);
}

proc makeNudgeMap(e: Expr) {
    var fvs = e.freeVars();
    var m = new map(string,real);
    for v in fvs {
        m[v] = e.grad(v);
    }
    return m;
}

proc cnst(x: real) {
    return new owned Constant(x): owned Expr;
}
 
proc main() {
    var a = lit("a",2); 
    var b = lit("b",3); 
    var c = a + b;
    c.showGradient("c");
    // var litA = lit("a",2);
    // var c = new AddExpr(litA,lit("b",3));// a + b; 

    var x = lit("x",4);
    var y = lit("y",10);
    var z = x * y;
    z.showGradient("z");

    var p = lit("p", 6);
    var q = lit("q", 2);
    var r = p / q;
    r.showGradient("r");

    var v = lit("v",3);
    var u = v ** 3;
    u.showGradient("u");

    var f = z * r * u;
    f.showGradient("f");

    // but I cannot do
    var g = x + f;
    // since x was consumed by `z = x * f`
}

