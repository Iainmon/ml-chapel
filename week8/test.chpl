import Tensor as tn;
use Tensor;



proc test(t: Tensor) {
    return t;
}
proc main() {
    var t = tn.zeros(3,3);
    writeln(test(t));
}