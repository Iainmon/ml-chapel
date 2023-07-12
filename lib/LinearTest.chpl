import Linear as lina;


proc main() {
    const v = new lina.Vector([1.0,2.0,3.0]);
    const m = lina.eye(3);
    const w = m * v;
    writeln(w);
}