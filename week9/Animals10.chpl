
import IO;
import BinaryIO;

import Tensor as tn;
use Tensor;

config const dataPath = "lib/animals-10/export/";

const categories = [
    // "chicken",
    "spider",
    // "cat",
    // "butterfly",
    "cow",
    // "horse",
    // "dog",
    // "sheep",
    "elephant",
    // "squirrel"
];

proc labelIdx(name: string): int {
    for i in categories.domain {
        if categories[i] == name {
            return i;
        }
    }

    tn.err("unknown category: ", name);
    return -1;
}

proc loadCategory(name: string): Tensor(4) {
    const path = dataPath + name + ".bin";
    var file = IO.open(path, IO.ioMode.rw);
    var deserializer = new BinaryIO.BinaryDeserializer(IO.ioendian.little);
    var fr = file.reader(deserializer=deserializer);

    var data = tn.zeros(0,0,0,0);
    data.read(fr);
    file.close();
    return data;
}

proc loadAll() {
    var catDom: domain(string);
    catDom += categories;
    var data: [catDom] Tensor(4);

    coforall name in categories {
        data[name] = loadCategory(name);
        writeln("[loaded ", dataPath, name, ".bin" ,"]");
    }
    return data;
}

iter loadAllIter() {
    for name in categories {
        yield (name,loadCategory(name));
        writeln("[loaded ", dataPath, name, ".bin" ,"]");
    }
}

iter loadAllIter(max: int) {
    for name in categories {
        var t = loadCategory(name);
        var (n,h,w,c) = t.shape;
        if n > max {
            t = new Tensor(t[0..#max, .., .., ..]);
            n = max;
        }

        for i in 0..#(n) {
            const im = new Tensor(t[i, .., .., ..]);
            yield (name,im);
        }
        writeln("[loaded ", dataPath, name, ".bin" ,"]");
    }
}

proc main() {
    // const t = loadCategory("chicken");
    // writeln(t.shape);
    for (name, data) in loadAllIter() {
        writeln(name, " ", data.shape);
    }

}