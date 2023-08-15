import Tensor as tn;
use Tensor;

import Time;

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
    /*var t = tn.zeros(2,2);
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
    }*/
    // const n = 100000000;
    // var a = [0,-1];
    // for i in 0..n do
    //     a[0] += 1;
    // writeln("serial \t\t", a);

    // a[0] = 0;
    // forall i in 0..n do
    //     a[0] += 1;
    // writeln("forall \t\t", a);

    // a[0] = 0;
    // forall i in 0..n with (+ reduce a) do
    //     a[0] += 1;
    // writeln("forall + reduce ", a);

    // a[0] = 0;
    // forall i in 0..n with (ref a) do
    //     a[0] += 1;
    // writeln("forall ref \t", a);

    // a[0] = 0;
    // foreach i in 0..n do
    //     a[0] += 1;
    // writeln("foreach \t", a);

    // {
    //     var n = 10000;
    //     var m = 50000;
    //     var t = tn.zeros(n,m);

    //     var st = new Time.stopwatch();
    //     st.start();

    //     const a = t.transpose();

    //     const tm = st.elapsed();
    //     writeln("out shape: ", a.shape);
    //     writeln("time: ", tm);
    // }

    // {
    //     var n = 10000;
    //     var m = 50000;
    //     var t1 = tn.zeros(n,m);
    //     var t2 = tn.zeros(n,m);

    //     t1.data = 3;

    //     var st = new Time.stopwatch();
    //     st.start();

    //     const a = t1 + t2;

    //     const tm = st.elapsed();
    //     writeln("out shape: ", a.shape);
    //     writeln("time: ", tm);
    // }

    {
        var m = 100000;
        var n = 130000;
        var p = 200;
        // var t1 = tn.randn(m,n);
        // var t2 = tn.randn(n,p);
        var v = tn.randn(n);
        writeln("done filling.");


        var st = new Time.stopwatch();
        st.start();

        // const M = t1 * t2;
        const w = v * tn.zeros(1,p);

        const tm = st.elapsed();
        writeln("out shape: ", w.shape, " ", w.shape);
        writeln("time: ", tm);
    }

}



