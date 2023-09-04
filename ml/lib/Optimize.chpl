
module Optimize {
    import Tensor as tn;
    use Tensor only Tensor;

    import Chai as ch;


    class SGD {
        var learnRate: real;
        var momentum: real;
        var decay: real;

        var iterations: int = 0;

        var velocitiesDomain = {0..#0};
        var velocities: [velocitiesDomain] Tensor(1);

        var uninitialized = true;

        proc init(learnRate: real, momentum: real = 0.0, decay: real = 0.0) {
            this.learnRate = learnRate;
            this.momentum = momentum;
            this.decay = decay;
        }

        proc initialize(ref network: ch.Network) {
            var velocityDomainSize = 0;
            for param n in 0..#(network.layers.size) {
                ref layer = network.layers[n];
                layer.populateParameters();
                var pmgr = layer.parameters;
                for (key, prm, grad) in pmgr do
                    velocityDomainSize += 1;
            }
            velocitiesDomain = {0..#velocityDomainSize};

            var i = 0;
            for param n in 0..#(network.layers.size) {
                ref layer = network.layers[n];
                layer.populateParameters();
                var pmgr = layer.parameters;
                for (key, prm, grad) in pmgr {
                    velocities[i] = prm.tensor;
                    velocities[i].data = 0.0;
                    i += 1;
                }
            }
            uninitialized = false;
        }

        iter parameters(ref network: ch.Network) {
            if uninitialized then initialize(network);
            var i = 0;
            for param n in 0..#(network.layers.size) {
                ref layer = network.layers[n];
                layer.populateParameters();
                var pmgr = layer.parameters;
                for (key, prm, grad) in pmgr {
                    yield (i, prm, grad);
                    i += 1;
                    // prm.tensor -= learnRate * grad.tensor;
                }
                layer.optimizeParameters();
            }
        }

        proc step(ref network: ch.Network) {
            // const lr = learnRate * (1.0 / (1.0 + decay * iterations));
            // for (i, prm, grad) in parameters(network) {
            //     velocities[i] = momentum * velocities[i] - lr * grad.tensor;
            //     grad.tensor = velocities[i];
            //     prm.tensor -= learnRate * grad.tensor;
            // }

            for (i, prm, grad) in parameters(network) {
                var gt = grad.tensor;
                var pt = prm.tensor;

                if decay > 0.0 then
                    gt += decay * pt;

                const lr = learnRate * (1.0 / (1.0 + decay * iterations));

                if iterations > 0 then
                    velocities[i] = momentum * velocities[i] + lr * gt;
                // else
                    // velocities[i] = gt;
                

                gt = velocities[i];
                grad.tensor = gt;
                prm.tensor -= learnRate * gt;
            }

            iterations += 1;

        }


    }

}