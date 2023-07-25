


record Tensor {
  param rank: int;

  var d: domain(rank);
  var a: [d] real;

  proc init(param rank) {

    if rank == 1 {
       //
    } else if rank == 2 {

    }
  }
}

record Network {

  param layerDom: domain(1);
  param layers: [d] owned Layer;

  proc feedForward(input): Tensor(?) {

    var x = input;
    for param l in layers {
      x = l.forwardProp(x);
    }

    select x.rank {
      when 1 do // ... ;
      when 2 {

      }
      otherwise do // ...
    }

  }

}

class CnnLayer: Layer {

  override dimIn(): param int do return 2;

  override proc forwardProp(x: Tensor(2)): Tensor(2) {
    // ...
  }
}

class MaxPool: Layer {

  override dimIn(): param int do return 2;

  override proc forwardProp(x: Tensor(2)): Tensor(1) {
    // ...
  }
}

class Dense: Layer {

  override dimIn(): param int do return 1;

  override proc forwardProp(x: Tensor(1)): Tensor(1) {
    // ...
  }
}

class Layer {

  proc dimIn() param : int {

  }

  proc forwardProp(x: Tensor(?)): Tensor(?) {
    // ...
  }
}
