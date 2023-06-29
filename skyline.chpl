config const n = 10;
var r = 1..n;

proc makeArray(i:int) {
  var A:[1..i] int;
  A[1] = i;
  return A;
}

// AA is an array of arrays
var AA = for i in r do makeArray(i);
for subarray in AA {
  writeln("subarray ", subarray);
  writeln("subarray domain ", subarray.domain);
  var B:subarray.type;
  writeln("subarray (B) type domain ", B.domain);
}

writeln("AA domain ", AA.domain);
var element:AA.eltType;
writeln("AA (element) eltType domain ", element.domain);
writeln("AA Type: ", AA.type:string);