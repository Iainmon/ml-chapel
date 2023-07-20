use Map;


var ages = [
    ("Iain",21),
    ("Kassia",20),
    ("Nick",23),
    ("Saul", 82),
    ("Martin", 55)
];

proc map.init(entries: [?d] ?t) where d.rank == 1 {
    type kt = entries.first[0].type;
    type vt = entries.first[1].type;
    this.init(kt,vt);
    for (k,v) in entries {
        this[k] = v;
    }
}

var ageMap = new map(ages);





proc foo(entries: [?d] ?t) {
    type t1 = entries.first[0].type;
    writeln(t:string, t1:string);
}

foo(ages);

writeln(ageMap);