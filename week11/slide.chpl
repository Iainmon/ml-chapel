

// Processes single input vector
proc forwardProp(input: Tensor(1)): Tensor(1) {
    return (weights * input) + bias;
}

// Processes a batch of input vectors in parallel
proc forwardPropBatch(batch: [?dom] Tensor(1)): [] Tensor(1) {
    var activations: [dom] Tensor(1);
    forall i in dom do
        activations[i] = forwardProp(batch[i]);
    return activations;
}