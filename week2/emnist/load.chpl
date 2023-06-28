

use IO;
use BinaryIO;


const fileName = "emnist/data/train-images-idx3-ubyte";

var deserializer = new BinaryDeserializer(ioendian.big);

var fr = openReader(fileName, deserializer=deserializer);

var magicNumber = fr.read(int(32));
writeln("Magic Number: 2051 = ",magicNumber);

var imageCount = fr.read(int(32));
writeln("Images: ", imageCount);

var rowCount = fr.read(int(32));
var columnCount = fr.read(int(32));

writeln("Dimensions: ", rowCount, " by ", columnCount);


var imageDomain = {0..#rowCount, 0..#columnCount};

proc readImage() {
    var raw: [imageDomain] uint(8);
    for i in 0..<rowCount {
        for j in 0..<columnCount {
            raw[i,j] = fr.read(uint(8));
        }
    }
    var image: [imageDomain] real = raw / 255.0;
    return image;
}

var im1 = readImage();

writeln(im1);

var images: [0..#(imageCount - 1)] [imageDomain] real;

for i in images.domain {
    images[i] = readImage();
    writeln("Image ", i);
}

