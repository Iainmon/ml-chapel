import Linear as lina;
import Chai as chai;
import IO;
import MNIST;

config const modelFile = "/Users/iainmoncrief/Documents/Sandbox/models/mnist.classifier.model";

config const testFile = "/Users/iainmoncrief/Documents/Processing/sketch_230720a/digit.txt"; // "digit.txt";

config const normalize = false;

proc loadImageBitmap(fn: string): [0..#28,0..#28] real {
    var file = IO.open(fn, IO.ioMode.r);
    var fr = file.reader();

    var bitmap: [0..#28,0..#28] real;
    for i in bitmap.domain {
        try {
            const pixel = fr.read(uint(8));
            bitmap[i] = pixel:real;
        } catch e: Error {
            halt(e);
        }
    }
    MNIST.printImage(bitmap);
    return bitmap;
}
proc vectorizeBitmap(bitmap: [0..#28,0..#28] real): [0..#784] real {
    var vector: [0..#(28 * 28)] real;
    for (m,n) in bitmap.domain {
        vector[m * 28 + n] = bitmap[m,n];
    }
    return vector;
}

proc main() {
    writeln("Loading image...");
    var image = new lina.Vector(vectorizeBitmap(loadImageBitmap(testFile)));
    
    if normalize then
        image = image.normalize();

    writeln("Loading model...");
    const model = chai.loadModel(modelFile);
    writeln("Model loaded.");

    writeln("Predicting...");
    const amplitudes = model.feedForward(image);
    writeln(amplitudes);

    const prediction = lina.argmax(amplitudes);
    writeln("Prediction: ", prediction);

}

