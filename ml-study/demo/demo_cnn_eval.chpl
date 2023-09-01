import Chai as chai;
import Tensor as tn;
use Tensor;
import Math;
import MNIST;
import IO;


// var net = new chai.Network(
//     (
//         new chai.Conv(1,16,5,stride=2),
//         new chai.Conv(16,64,3),
//         new chai.MaxPool(),
//         new chai.SoftMax(10)
//     )
// );

var net = new chai.Network(
    (
        new chai.Conv(1,8,7,stride=1),
        new chai.Conv(8,12,5),
        new chai.MaxPool(),
        new chai.SoftMax(10)
    )
);

// var net = new chai.Network(
//     (
//         new chai.Conv(1,8,7,stride=1),
//         new chai.Conv(8,12,5),
//         new chai.Conv(12,16,3),
//         new chai.MaxPool(),
//         new chai.SoftMax(10)
//     )
// );

const modelFile = "ml-study/lib/models/mnist" + net.signature() + ".model";

config const testFile = "/Users/iainmoncrief/Documents/Processing/sketch_230720a/digit.txt"; // "digit.txt";

config const normalize = false;
config const recenter = false;

config const run = false;

proc loadImageBitmap(fn: string): [0..#28,0..#28] real {
    var file = IO.open(fn, IO.ioMode.r);
    var fr = file.reader();

    var bitmap: [0..#28,0..#28] real;
    for i in bitmap.domain {
        try {
            const pixel = fr.read(uint(8));
            bitmap[i] = pixel;
        } catch e: Error {
            halt(e);
        }
    }
    MNIST.printImage(bitmap);
    bitmap /= 255.0;
    return bitmap;
}


// net.load(modelFile);
net.load(modelFile);

writeln("Loading image...");
var im = loadImageBitmap(testFile);

if recenter then im -= 0.5;

var image = (new Tensor(im)).reshape(28,28,1);

if normalize then
    image = image.normalize();

writeln("Predicting...");
const amplitudes = net.forwardProp(image);
writeln(amplitudes);

const prediction = tn.argmax(amplitudes.data);
writeln("Prediction: ", prediction);
