

use LinearAlgebra;
use Math;

var A = Matrix([0.0, 1.0, 1.0],
               [1.0, 0.0, 1.0],
               [1.0, 1.0, 0.0]);
var I = eye(3,3);
var B = A + I;

// writeln(B);

// proc applyLayer(weights: [?d] real, bias: real, inputs: [d] real) {}

proc applyLayerLA(weights: [?d] ?t, bias: t, inputs: [d] t) {
    var sum = dot(weights,inputs) + bias;
    var activation = tanh(sum);
    return activation;
}

var ws = Vector([1.0, 1.0, 0.0],eltType=real);
var x = Vector([1.0, 2.0, 3.0],eltType=real);
var bias = 0.7;
writeln(ws.type:string);
var result = applyLayerLA(ws,bias,x);
writeln(result);