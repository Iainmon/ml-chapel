
// Is this the best way to initialize arrays in classes?
class C {
    var arrayDom = {0..1};
    var array: [arrayDom] int;
    proc init(array: [?d] int) {
        this.arrayDom = d;
        this.array = array;
    }
}

var c = new C([0, 1, 3, 4]);

writeln(c);


// How to initialize biases and weights in a neural network?

// class Network {
//     var layerSizesDomain = {0..2};
//     var layerSizes: [layerSizesDomain] int;
//     var numLayers: int;

//     var biasesDomain = {0..2};
//     var biases: [biasesDomain] real;

//     var weightsDomain = {0..2};
//     var weights: [1..0] [1..0, 1..0] real; // what should the second domain be?

//     proc init(layerSizes: [?d] int) {
//         layerSizesDomain = d;
//         this.layerSizes = layerSizes;
//         this.numLayers = layerSizes.size;

//         // How to initialize biases and weights?

//         // Goal:
//         // biases = [y in layerSizes[1..]] 0:real;

//         var biases_ = [y in layerSizes[1..]] 0:real;
//         biasesDomain = biases_.domain;
//         biases = biases_;


//         // Goal:
//         // weights = [(x, y) in zip(layerSizes[0..#(numLayers-1)], layerSizes[1..])] createRandomMatrix(y, x);

//         var weights_ = [];
//         for (x, y) in zip(layerSizes[0..#(numLayers-1)], layerSizes[1..]) do
//             weights_.append(createRandomMatrix(y, x));
//         weightsDomain = weights_.domain;
//         weights = weights_;
//     }
// }

// var net = new Network([2, 3, 1]);


// var xs = [1,2,3];

// var ys = [x in xs] [1..#x];

// writeln(ys);


// class MatrixWraper {
//     var matrixDomain: domain(2,int,false); //= {0..2, 0..2};
//     var matrix: [1..0, 1..0] real;
//     proc init(matrix: [?d] real) where d.rank == 2 {
//         matrixDomain = d;
//         this.matrix = matrix;
//         var (m,n) = matrix.domain.shape;
//         // d.dim(0).size
//     }
//     proc shape {
//         return this.matrix.domain.shape;
//     }
// }

var xs = [1,2,3,4,5,6];


proc access(xs: [?d] ?eltType, i: int) {
    return xs[d.orderToIndex((d.size + i) % d.size)];
}
writeln(xs[xs.domain.orderToIndex(xs.domain.size - 1)]);
writeln(access(xs,10));

for i in 2..3 do
    writeln(i);