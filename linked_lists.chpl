private use List;
import List;




// class Value {
//     var data: real;
//     var children = new list(Value); // = new list(Value);
//     var op: string;

//     proc init(data: real) {
//         this.data = data;
//         this.children = new list(Value);
//         this.op = '';

//     }
//     proc init(data: real, children: list(Value), op = '') {
//         this.data = data;
//         this.children = children;
//         this.op = op;
//     }

// }

// const v : Value = new Value(1);

// writeln(v);




class RoseTree {
    type t;
    var root: t;
    var children: list(owned RoseTree(t)) = new list(owned RoseTree(t));
}



type IntRT = RoseTree(int);

var t = new RoseTree(int,1);
writeln(t);

var t2 = new IntRT(1);
writeln(t2);




class LinkedList {
    type t;
    var value: t;
    var next: owned LinkedList(t)? = nil;
}

proc cons(x: ?t, in l: LinkedList(t)) {
    var head = new LinkedList(t,x,);
    return head;
}

var l1 = new LinkedList(int,1);
writeln(l1);

var l2 = cons(1,l1);
writeln(l2);




class Link {
    var value;
    var next: owned Link(value.type)? = nil;
    proc underlying() {
        return this.value.type;
    }
}

// var n = new Link(1);
// writeln(n.underlying());

// proc cons(l: LinkedList, ) {

// }

// proc append()



// proc link_cons(l: Link(t),x: t) {
//     return 1;
// }

// var n = new Link(1);
// writeln(n);
// writeln(link_cons(n,1));
