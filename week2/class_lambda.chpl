
class LambdaHolder {
    var myLambda: func(real,real);
    proc callme() {
        return myLambda(1.0);
    }
}


proc helloProc(x: real) { writeln("Hello"); return 0.0; }

var helloLambda = lambda(x: real) { writeln("Hello"); return 0.0; };

var helloAnonProc = proc(x: real) { writeln("Hello"); return 0.0; };

var lh = new LambdaHolder(helloLambda);
lh.callme();


writeln(helloProc.type:string, ", ", helloLambda.type:string);