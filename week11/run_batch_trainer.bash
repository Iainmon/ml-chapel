#!/bin/bash

mkdir -p batch_tests
mkdir -p batch_tests/output
mkdir -p batch_tests/output/python
mkdir -p batch_tests/output/chapel

# Run the python tests
# for data_size in 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300; do
for data_size in 20 40 60 80 100; do
    salloc --nodes=1 --partition=chapdl --exclusive srun ./mnist_trainer --dataSize=$data_size >> batch_tests/output/chapel/mnist_trainer_$data_size.csv &
done

jobs > batch_tests/jobs.txt