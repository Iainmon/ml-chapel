use Tensor, Network, LayerTypes;

proc main() {

  var c = new Conv(1, 2, 3);

  var fs: [c.filters.data.domain] real = 1;
  fs[0,0,0,..] = [1, 1, 1];
  fs[0,0,1,..] = [1, -8, 1];
  fs[0,0,2,..] = [1, 1, 1];

  fs[1,0,0,..] = [-1, 0, 1];
  fs[1,0,1,..] = [-1, 0, 1];
  fs[1,0,2,..] = [-1, 0, 1];
  c.setFilter(fs);

  //test normal
  var t1 = Tensor.ones(1, 10, 10);
  writeln(t1);

  var t2 = c.forwardProp(t1);
  writeln(t2);

  var t3 = c.backwardProp(t2, t1);
  writeln(t3);

  // // test bulk
  // var T1 = Tensor.randn(20, 3, 12, 12);
  // var T2 = c.forwardPropBulk(T1);
  // var T3 = c.backwardPropBulk(T2, T1);

  // writeln(T3[0, .., .., ..]);
}
