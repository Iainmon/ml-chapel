import Tensor as tn;
use Tensor;



proc test(t: Tensor) {
    return t;
}
proc main() {
    var t = tn.zeros(2,2);
    t[0,0] = 1;
    t[0,1] = 2;
    t[1,0] = 3;
    t[1,1] = 4;

    for i in t.data.domain {
        writeln(t.data[i]);
    }
    writeln(test(t));
}

