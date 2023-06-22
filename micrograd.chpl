
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

    proc nudge(d: map(string,real)) {
        
    }

    operator +(a: Expr, b: Expr) { return (new shared AddExpr(a,b)): Expr; }
    operator -(a: Expr, b: Expr) { return (new shared SubExpr(a,b)): Expr; }
    operator *(a: Expr, b: Expr) { return (new shared MulExpr(a,b)): Expr; }
    operator /(a: Expr, b: Expr) { return (new shared DivExpr(a,b)): Expr; }
    operator **(a: Expr, c: real) { return (new shared ExpExpr(a,c)): Expr; }
    proc relu() { return (new shared ReluExpr(this)): Expr; }
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

    override proc freeVars() {
        return new set(string,[this.name]);
    }

    override proc value() {
        return this.base;
    }

    override proc grad(name: string): real {
        if this.name == name {
            return 1;
        }
        return 0;
    }
    override proc nudge(d: map(string,real)) {
        var diff = d[this.name];
        var learnRate = 0.3;
        this.base +=  -diff * learnRate;
    }
}

proc lit(name: string, value: real): Expr {
    return (new shared Literal(name,value)): Expr;
}

class Constant: Expr {
    var base: real;
    override proc value() { return this.base; }
    proc init() { this.base = 0;}
    proc init(base: real) { this.base = base; }
}

enum Op { plus,minus,times,div }

class Compound: Expr {
    var children: list(shared Expr);

    proc init(children: list(shared Expr)) { this.children = children; }
    proc init() { this.children = new list(shared Expr); }

    proc binOpInit(a: shared Expr, b: shared Expr) {
        this.children = new list(shared Expr);
        this.children.append(a);
        this.children.append(b);
    }

    override proc freeVars() {
        // return + reduce this.children.these().freeVars();
        var fvs = new set(string);
        for c in this.children {
            fvs += c.freeVars();
        }
        return fvs;
    }

    override proc nudge(d: map(string,real)) {
        for c in this.children {
            c.nudge(d);
        }
    }

    proc lhs() { return this.children[0]; }
    proc rhs() { return this.children[1]; }
}

class AddExpr: Compound {
    proc init(a: Expr, b: Expr) { this.binOpInit(a,b); }
    override proc value() { return this.lhs().value() + this.rhs().value(); }
    override proc grad(name: string) {
        return this.lhs().grad(name) + this.rhs().grad(name);
    }
}
class SubExpr: Compound {
    proc init(a: Expr, b: Expr) { this.binOpInit(a,b); }
    override proc value() { return this.lhs().value() - this.rhs().value(); }
    override proc grad(name: string) {
        return this.lhs().grad(name) - this.rhs().grad(name);
    }
}
class MulExpr: Compound {
    proc init(a: Expr, b: Expr) { this.binOpInit(a,b); }
    override proc value() { return this.lhs().value() * this.rhs().value(); }
    override proc grad(name: string) {
        return this.lhs().grad(name) * this.rhs().value() + this.rhs().grad(name) * this.lhs().value();
    }
}
class DivExpr: Compound {
    proc init(a: Expr, b: Expr) { this.binOpInit(a,b); }
    override proc value() { return this.lhs().value() / this.rhs().value(); }
    override proc grad(name: string) {
        var f = this.lhs();
        var fx = f.value();
        var df = f.grad(name);

        var g = this.rhs();
        var gx = g.value();
        var dg = g.grad(name);

        return (df * gx - dg * fx) / (gx ** 2);
    }
}


class ExpExpr: Compound {
    proc init(a: Expr, c: real) {
        var cexpr = new shared Constant(c);
        this.binOpInit(a,cexpr: Expr);
    }
    override proc value() { return this.lhs().value() ** this.rhs().value(); }
    override proc grad(name: string) {
        var c = this.rhs().value();
        return c * (this.lhs().value() ** (c - 1)) * this.lhs().grad(name);
    }
}

class ReluExpr: Expr {
    var x: Expr;
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

proc relu(e: shared Expr) {
    return new shared ReluExpr(e);
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
    return new shared Constant(x): Expr;
}


// Some examples

/*
var a: Expr = new shared Literal("a",2); // lit("a",2);
var b: Expr = new shared Literal("b",3); // lit("b",3);
var c: Expr = a + b; // new shared AddExpr(a, b);
c.showGradient("c");

var x: Expr = new shared Literal("x", 4);
var y: Expr = new shared Literal("y", 10);
var z: Expr = x * y; // new shared MulExpr(x, y);
z.showGradient("z");

var p = lit("p", 6);
var q = lit("q", 2);
var r = p / q;
r.showGradient("r");

var v: Expr = new shared Literal("v", 3);
var u = v ** 3; // new shared ExpExpr(v, 10);
u.showGradient("u");


var f = c * z * u;
f.showGradient("f");

*/

