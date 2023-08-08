import Torch as torch;
import Tensor as tn;
use Tensor;
import Math;
import MNIST;
import IO;

config const modelFile = "models/cnn/epoch_0_mnist.cnn.model";

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

// var net = new torch.Network(
//     (
//         new torch.Conv(16),
//         new torch.MaxPool(),
//         new torch.SoftMax(13 * 13 * 16,10)
//     )
// );

// var net = new torch.Network(
//     (
//         new torch.Conv(1,20,3),
//         new torch.MaxPool(),
//         // new torch.SoftMax(13 * 13 * 8,10)
//         new torch.Conv(20,10,3),
//         new torch.MaxPool(),
//         new torch.SoftMax(5 * 5 * 10,10)
//     )
// );

var net = new torch.Network(
    (
        new torch.Conv(1,8,7),
        new torch.Conv(8,12,5),
        new torch.MaxPool(),
        new torch.SoftMax(10)
    )
);

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
