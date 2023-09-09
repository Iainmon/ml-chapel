import Tensor as tn;
use Tensor only Tensor;
use CTypes;
use ChapelArray;
use ChapelBase;

var t = new Tensor({0..#3,0..#2});

for (i,j) in t.domain {
    t[i,j] = i*2+j;
}

writeln(t);

var ptr = c_ptrTo(t.data);
writeln(ptr,ptr.type:string);
writeln(ptr.eltType:string);

var ptr2 = _ddata_allocate(real,t.size);


writeln(ptr2,ptr2.type:string);


// var ptr3 = c_ptrTo(t);

// t[1,1] = 5.0;

// writeln(ptr3,ptr3.type:string);
// writeln(ptr3.eltType:string);
// ref t_3 = ptr3.deref();
// writeln(t_3);
// t_3[1,1] = 6.0;
// writeln(t);

// writeln(ptr2.deref());
// var ptr2_real = ptr2: c_ptr(real);
// writeln(c_offsetof(Tensor(2), "rank"));

// for i in 0..#10 {
//     writeln(ptr2_real[i]);
// }

// var ptr4: c_ptr(void) = c_addrOf(t.data);
// writeln(ptr4,ptr4.type:string);
// var ptr4_rankptr = ptr4: c_array(real);
// // for i in 0..5 {
// //     writeln(ptr4_rankptr[i]);
// // }
// writeln(ptr4_rankptr);

record NDArrayRef {
    var rank: int;
    var shapeSize: int;
    var shape: c_ptr(int);
    var data: c_ptr(real);
    proc init(ref array: [?d] real) {
        rank = d.rank;
        shapeSize = array.shape.size;
        shape = allocate(int, shapeSize: c_size_t);
        var arrayRef: c_ptr(real) = c_ptrTo(array);
        data = arrayRef;
    }
    proc init(ref t: Tensor(?)) {
        this.init(t.data);
    }
}

var aref1 = new NDArrayRef(t.data);
writeln(aref1.data[1]);

record TensorRef {
    var rank: int;
    var shapeSize: int;
    var shape: c_ptr(int);
    var data: c_ptr(real);
    proc init(ref t: Tensor(?)) {
        rank = t.rank;
        shapeSize = t.shape.size;
        shape = allocate(int, shapeSize: c_size_t);
        var arrayRef: c_ptr(real) = c_ptrTo(t.data);
        data = arrayRef;
    }
}

record Errase {
    var rank: int;
    var ptr: c_ptr(void);
    proc init(ref t: Tensor(?)) {
        rank = t.rank;  
        ptr = c_ptrTo(t): c_ptr(void);
    }

}

writeln(c_offsetof(Errase,"rank"));
writeln(c_offsetof(Errase,"ptr"));
var xs = [1.0,2.0,3.0];
var a: _array = xs;
var ys = [9.0,8.0,7.0];
var b: _array = ys;

writeln(a.type:string);
writeln(a._value.type:string);
var c = a._value;
var d = b._value;

// b[1] = 5.0;
writeln(a);
writeln(b);

// var z = t.data;
// var zz = z : _ddata(real);
// writeln(zz);

// writeln(a);
// writeln(b);
// writeln(c_offsetof(a.type,"_instance"));