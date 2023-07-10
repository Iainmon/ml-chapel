module Graph {

    import Set;
    import Map;

    iter cartesian(a,b) {
        for x in a {
            for y in b {
                yield (x,y);
            }
        }
    }

    record Bijection {
        type domainType;
        type codomainType;

        // var forwardD: domain(domainType)
        // var forward: [forwardD] int;
        var forward: Map.map(domainType,codomainType) = new Map.map(domainType,codomainType);
        var backward: Map.map(codomainType,domainType) = new Map.map(codomainType,domainType);

        proc inverse() {
            var inv = new Bijection(codomainType,domainType);
            inv.forward = backward;
            inv.backward = forward;
            return inv;
        }

        iter these() const ref  {
            for (k,v) in forward.items() {
                yield (k,v);
            }
        }

        proc insert(k: domainType, v: codomainType) {
            forward.addOrSet(k,v);
            backward.addOrSet(v,k);
        }

        proc remove(k: domainType) {
            var v = forward[k];
            forward.remove(k);
            backward.remove(v);
        }
        proc remove(v: codomainType) {
            var k = backward[v];
            forward.remove(k);
            backward.remove(v);
        }

        iter dom {
            for k in forward.keys() {
                yield k;
            }
        }

        iter cod {
            for k in backward.keys() {
                yield k;
            }
        }

        proc this(k: domainType) ref throws {
            return forward[k];
        }
        proc this(k: codomainType) ref throws {
            return backward[k];
        }

    }

    record graph {
        type eltType;

        var vertexHash: Bijection(eltType,int) = new Bijection(eltType,int);
        var vertices: Set.set(eltType) = new Set.set(eltType);
        var neighbors: Map.map(int,Set.set(int)) = new Map.map(int,Set.set(int));

        proc addVertex(v: eltType) {
            if (vertices.contains(v)) {
                return;
            }
            vertices.add(v);
            var i = vertexHash.forward.size;
            vertexHash.insert(v,i);
            neighbors.addOrSet(i,new Set.set(int));
        }

        proc addEdge(v1: eltType, v2: eltType) {
            addVertex(v1);
            addVertex(v2);
            
            var i1 = try! vertexHash[v1];
            var i2 = try! vertexHash[v2];
            try! neighbors[i1].add(i2);
            // neighbors[i2].add(i1);
        }

        proc removeVertex(v: eltType) {
            if (!vertices.contains(v)) {
                return;
            }
            var i = vertexHash[v];
            vertices.remove(v);
            vertexHash.remove(v);
            neighbors.remove(i);
            for (k,s) in neighbors.items() {
                s.remove(i);
            }
        }

        proc removeEdge(v1: eltType, v2: eltType) {
            if (!vertices.contains(v1) || !vertices.contains(v2)) {
                return;
            }
            var i1 = vertexHash[v1];
            var i2 = vertexHash[v2];
            neighbors[i1].remove(i2);
            neighbors[i2].remove(i1);
        }

        iter neighborhood(v: eltType) ref {
            var i = vertexHash[v];
            for j in neighbors[i] {
                yield vertexHash[j];
            }
        }

        proc this(v: eltType) ref : Set.set(eltType) {
            var i = vertexHash[v];
            var s = new Set.set(eltType);
            for n in neighborhood(v) {
                s.insert(n);
            }
            return s;
        }

        proc slice(v: eltType, n: int = 0) {
            if n == 0 then
                return new Set.set([v]);
            if n == 1 then 
                return new Set.set(neighborhood(v));
            var acc = new Set.set(eltType);
            for u in neighborhood(v) {
                acc += slice(u,n-1);
            }
            return acc;
        }

        proc bleed(v: eltType, n: int = 0) {
            var acc = new Set.set(eltType);
            for i in 0..n {
                acc += slice(v,i);
            }
            return acc;
        }

        proc isCyclic(): bool {
            return true;
        }

        proc adjacencyMatrix() {
            var n = vertices.size;
            var m: [0..#n,0..#n] bool = false;
            for i in 0..#n {
                for j in 0..#n {
                    if (try! neighbors[i]).contains(j) {
                        m[i,j] = true;
                    }
                }
            }
            return m;
        }

        proc labeldAdjacencyMatrix() {
            var m = adjacencyMatrix();
            var labelDomain: domain((eltType,eltType)) = cartesian(vertices,vertices);
            var lm: [labelDomain] bool = false;
            for (i,j) in labelDomain {
                lm[(i,j)] = m[try! vertexHash[i],try! vertexHash[j]];
            }
            return lm;
        }
    }



}

proc main() {
    var G = new Graph.graph(eltType=string);
    G.addVertex("Iain");
    G.addVertex("Kassia");
    G.addVertex("Ella");
    G.addEdge("Iain","Kassia");
    G.addEdge("Iain","Ella");
    writeln(G.adjacencyMatrix());
    writeln(G.labeldAdjacencyMatrix());
}