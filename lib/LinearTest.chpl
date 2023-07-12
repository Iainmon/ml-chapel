import Linear as lina;


proc main() {
    {
        const v = new lina.Vector([1.0,2.0,3.0]);
        const m = lina.eye(3);
        const w = m * v;
        writeln(w);
    }

    {
        const m = lina.matrixFromRows([1,2,3],real);
        const v = new lina.Vector(m);
        writeln(v);
    }

    {
        const m = lina.matrixFromColumns([1,2,3],real);
        const v = new lina.Vector(m);
        writeln(v);
    }

    {
        const v1 = new lina.Vector([1,2,3]);
        const v2 = new lina.Vector([1,2,3]);
        const m = new lina.Matrix(v2);
        const o = v1 * m.transpose();
        writeln(o);
    }

    {
        const v = new lina.Vector([1,2,3]);
        writeln(v);
        writeln(v.transpose());
    }
}