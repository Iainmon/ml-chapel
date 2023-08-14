import Tensor as tn;
use Tensor;



proc test(t: Tensor) {
    return t;
}
iter cartesian(X,Y) {
    for x in X {
        for y in Y {
            yield (x,y);
        }
    }
}
iter cartesian(param tag: iterKind,X,Y) where tag == iterKind.standalone {
    forall x in X {
        forall y in Y {
            yield (x,y);
        }
    }
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

    forall (x,y) in cartesian( 0..10,0..10) {
        writeln((x,y));
    }
}



