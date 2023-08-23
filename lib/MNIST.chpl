

use IO;
use BinaryIO;

proc loadImages(num: int, fileName: string = "week2/emnist/data/train-images-idx3-ubyte") {

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

    var images: [0..#num] [imageDomain] real(64);

    for i in images.domain {
        images[i] = readImage();
        // writeln("Image ", i);
    }
    return images;

}


proc loadLabels(num: int, fileName: string = "week2/emnist/data/train-labels-idx1-ubyte") {

    var deserializer = new BinaryDeserializer(ioendian.big);

    var fr = openReader(fileName, deserializer=deserializer);

    var magicNumber = fr.read(int(32));
    writeln("Magic Number: 2049 = ",magicNumber);

    var labelCount = fr.read(int(32));
    writeln("Labels: ", labelCount);

    var labels: [0..#num] int;

    for i in labels.domain {
        labels[i] = fr.read(int(8));
        // writeln("Label ", i, ": ", labels[i]);
    }

    var formattedLabels: [0..#num] [0..#10] real(64);

    for i in labels.domain {
        for j in 0..<10 {
            formattedLabels[i][j] = if labels[i] == j then 1.0 else 0.0;
        }
    }

    return (labels, formattedLabels);
}

proc printImage(image: [0..#28, 0..#28] real) {
    for i in 0..<28 {
        for j in 0..<28 {
            if image[i,j] > 0.5 then {
                write("#");
            } else {
                write(" ");
            }
        }
        writeln();
    }
}