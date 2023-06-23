

use micrograd;


proc supportsPlus(c: class )

// operator +(x: owned class?, y: x.type): x.type {
//     if x == nil && y == nil then return nil;
//     if x == nil then return new x.type(y!);
//     if y == nil then return new x.type(x!);
//     return (x! + y!) : x.type;
// }
operator +(const ref x: (?c)?, const ref y: c?): c? where supportsPlus(c) {
  if x == nil && y == nil then return nil;
  if x == nil then return new c(y!); // invoke copy constructor 
  if y == nil then return new c(x!); // invoke copy constructor
  return x! + y!;
}

var cs = [i in 1..10] cnst(i);
writeln(cs);

writeln(+ reduce cs);

ref x = cs[1];
writeln(x);
writeln(x.type:string);
