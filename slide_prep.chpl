

class Expr {
    proc freeVars(): set(string);
    proc grad(name: string): real;
    proc value(): real;
    proc nudge(d: map(string,real)): void;
    operator +(a: Expr, b: Expr): Expr; // et al
    proc relu(): Expr;
}
class Constant: Expr { ... }
class Variable: Expr { ... }
class AddExpr: Expr { ... }

record Matrix {
    type eltType;
    var matrixDomain: domain(2,int);
    var underlyingMatrix: [matrixDomain] eltType; 
    ...
} // Similar for vectors

class Network {
    ...
    var layerSizes: [layerSizesDomain] int;
    var numLayers: int;
    var biases: [biasesDomain] lina.Vector(real(64));
    var weights: [weightsDomain] lina.Matrix(real(64));

    proc feedForward(input: Vector(real)): Vector(real);
}

var net = new Network([3,4,2]);


config const numImages = 8000;
var net = new Network([28 * 28, 200, 80, 10]);
