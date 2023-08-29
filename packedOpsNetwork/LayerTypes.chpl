module LayerTypes {
  use Tensor;

  record Conv {
    var filters: Tensor(4);

    const numOut: int;
    const numIn: int;
    const k: int;

    const kernelDom: domain(3);
    const kernelDomOut: domain(3);

    proc init(inChannels: int, outChannels: int, kernelSize: int = 3) {
      this.filters = Tensor.randn(outChannels, inChannels, kernelSize, kernelSize);
      this.numOut = outChannels;
      this.numIn = inChannels;
      this.k = kernelSize;
      this.kernelDom = {0..<inChannels, 0..<kernelSize, 0..<kernelSize};
      this.kernelDomOut = {0..<outChannels, 0..<kernelSize, 0..<kernelSize};
    }

    proc setFilter(f: [?d] real) where d.rank == 4 {
      this.filters.data = f;
    }

    /*
      s:
      - dim 0: in-channels
      - dim 1: x
      - dim 2: y

      returns:
      - dim 0: out-channels
      - dim 1: x-k
      - dim 2: y-k
    */
    proc forwardProp(s: Tensor(3)): Tensor(3) {
      const (nc, nx, ny) = s.shape;
      assert(
        nc == this.numIn,
        "Channel# mismatch: conv Layer expected " + this.numIn:string + " channels, got " + nc:string + " channels."
      );

      var convs: [0..#numOut, 0..<nx-k-1, 0..<ny-k-1] s.eltType;

      const imgInner = ({0..<nx, 0..<ny}).expand(-(k-1)),
            imageDom = {0..<numOut, imgInner.dim(0), imgInner.dim(1)},
            kOffset = if k%2 == 0 then k/2 else k/2 + 1;

      forall (cout, i, j) in imageDom {
        var sum = 0.0;
        for (cin, p, q) in this.kernelDom do
          sum += filters[cout, cin, p, q] * s[cin, i, j];

        convs[cout, i - kOffset, j - kOffset] = sum;
      }

      return new Tensor(convs);
    }

    /*
      samples:
      - dim 0: n images
      - dim 1: in-channels
      - dim 2: x
      - dim 3: y

      returns:
      - dim 0: n images
      - dim 1: out-channels
      - dim 2: x-k
      - dim 3: y-k
    */
    proc forwardPropBulk(samples: Tensor(4)): Tensor(4) {
      const (n, nc, nx, ny) = samples.shape;
      assert(
        nc == this.numIn,
        "Channel# mismatch: conv Layer expected " + this.numIn:string + " channels, got " + nc:string + " channels."
      );

      var convs: [0..#n, 0..#numOut, 0..<nx-k-1, 0..<ny-k-1] samples.eltType;

      const imgInner = ({0..<nx, 0..<ny}).expand(-(k-1)),
            imageDom = {0..<n, 0..<numOut, imgInner.dim(0), imgInner.dim(1)},
            kOffset = if k%2 == 0 then k/2 else k/2 + 1;

      forall (img, cout, i, j) in imageDom {
        var sum = 0.0;
        for (cin, p, q) in this.kernelDom do
          sum += filters[cout, cin, p, q] * samples[img, cin, i, j];

        convs[img, cout, i - kOffset, j - kOffset] = sum;
      }

      return new Tensor(convs);
    }

    /*
      grad:
      - dim 0: out-channels
      - dim 1: x-k
      - dim 2: y-k

      s:
      - dim 0: in-channels
      - dim 1: x
      - dim 2: y

      returns:
      - dim 0: in-channels
      - dim 1: x
      - dim 2: y
    */
    proc backwardProp(grad: Tensor(3), s: Tensor(3)): Tensor(3) {
      const (nc, nx, ny) = s.shape;
      assert(
        nc == this.numIn,
        "Channel# mismatch: conv Layer expected " + this.numIn:string + " channels, got " + nc:string + " channels."
      );

      const imageDom = ({0..<nx, 0..<ny}).expand(-(k-1)),
            kOffset = if k%2 == 0 then k/2 else k/2 + 1;

      // compute dLdF (to update Filters)
      var dLdF : [0..<numOut, 0..<numIn, 0..<k, 0..<k] s.eltType;

      forall cout in 0..<numOut {
        for (cin, p, q) in this.kernelDom {
          var sum = 0.0;
          forall (i, j) in imageDom with (+ reduce sum) do
            sum += s[cin, i, j] * grad[cout, i - kOffset, j - kOffset];

          dLdF[cout, cin, p, q] = sum;
        }
      }
      this.filters += new Tensor(dLdF);

      // compute dLdS (to pass to prev layer)
      var dLdS: [0..<numIn, 0..<nx, 0..<ny] s.eltType;

      forall cin in 0..<numIn {
        forall (i, j) in imageDom {
          var sum = 0.0;
          for (cout, p, q) in this.kernelDomOut do
            sum += filters[cout, cin, k-p-1, k-q-1] * s[cin, i, j];

          dLdS[cin, i, j] = sum;
        }
      }
      return new Tensor(dLdS);
    }

    /*
      grads:
      - dim 0: n images
      - dim 1: out-channels
      - dim 2: x-k
      - dim 3: y-k

      samples:
      - dim 0: n images
      - dim 1: in-channels
      - dim 2: x
      - dim 3: y

      returns:
      - dim 0: n images
      - dim 1: in-channels
      - dim 2: x
      - dim 3: y
    */
    proc backwardPropBulk(grads: Tensor(4), samples: Tensor(4)): Tensor(4) {
      const (n, nc, nx, ny) = samples.shape;
      assert(
        nc == this.numIn,
        "Channel# mismatch: conv Layer expected " + this.numIn:string + " channels, got " + nc:string + " channels."
      );

      const imgInner = ({0..<nx, 0..<ny}).expand(-(k-1)),
            imageDom = {0..<n, imgInner.dim(0), imgInner.dim(1)},
            kOffset = if k%2 == 0 then k/2 else k/2 + 1;

      // compute mean dLdF over samples (to update Filters)
      var dLdF_mean : [0..<numOut, 0..<numIn, 0..<k, 0..<k] samples.eltType;

      forall cout in 0..<this.numOut {
        for (cin, p, q) in this.kernelDom {
          var sum = 0.0;
          forall (img, i, j) in imageDom with (+ reduce sum) do
            sum += samples[img, cin, i, j] * grads[img, cout, i - kOffset, j - kOffset];

          dLdF_mean[cout, cin, p, q] = sum;
        }
      }
      dLdF_mean /= n;
      this.filters += new Tensor(dLdF_mean);

      // compute dLdS for each image (to pass to prev layer)
      var dLdS: [0..<n, 0..<numIn, 0..<nx, 0..<ny] samples.eltType;

      forall cin in 0..<numIn {
        forall (img, i, j) in imageDom {
          var sum = 0.0;
          for (cout, p, q) in this.kernelDomOut do
            sum += filters[cout, cin, k-p-1, k-q-1] * samples[img, cin, i, j];

          dLdS[img, cin, i, j] = sum;
        }
      }
      return new Tensor(dLdS);
    }
  }

}
