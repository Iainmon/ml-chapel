#!/bin/bash

# run `python3 trainer.py {data_size} True {hidden_layer_size}`, varying
# data_size = {20, 40, 60, 80, 100, 120, 140, 160, 180, 200} 
# and hidden_layer_size = {4, 8, 12, 16, 20, 24}. Then strip the last line in the output,
# which will be of the form `time: {num_seconds}`. Append the line `data_size, hidden_layer_size, num_seconds` to a 
# file called `speed_tests/trainer.py.csv`. Then do the same thing for 
# `./trainer --dataSize={data_size} --hiddenLayerSize={hidden_layer_size} --parallelize={parallelize}`, but
# with parallelize = {true, false}, and append to `speed_tests/trainer.chpl.csv`.

mkdir -p speed_tests

echo "data_size, hidden_layer_size, num_seconds" > speed_tests/trainer.py.csv
echo "data_size, hidden_layer_size, parallelize, num_seconds" > speed_tests/trainer.chpl.csv

# Run the python tests
for data_size in 20 40 60 80 100 120 140 160 180 200; do
    for hidden_layer_size in 4 8 12 16 20 24; do
        # Print indication of progress
        echo "Running data_size=$data_size, hidden_layer_size=$hidden_layer_size"

        # Run the command and capture its time
        time_output=$(python3.6 trainer.py $data_size True $hidden_layer_size | tail -n 1)

        # Extract num_seconds from the output
        num_seconds=$(echo $time_output | awk -F ': ' '{print $2}')

        # Append to the csv file
        echo "$data_size, $hidden_layer_size, $num_seconds" >> speed_tests/trainer.py.csv

        # Print elapsed time
        echo "Elapsed time: $num_seconds"
    done
done

# Run the chapel tests
for data_size in 20 40 60 80 100 120 140 160 180 200; do
    for hidden_layer_size in 4 8 12 16 20 24; do
        for parallelize in true false; do
            # Print indication of progress
            echo "Running data_size=$data_size, hidden_layer_size=$hidden_layer_size, parallelize=$parallelize"

            # Run the command and capture its time
            time_output=$(./trainer --dataSize=$data_size --hiddenLayerSize=$hidden_layer_size --parallelize=$parallelize | tail -n 1)

            # Extract num_seconds from the output
            num_seconds=$(echo $time_output | awk -F ': ' '{print $2}')

            # Append to the csv file
            echo "$data_size, $hidden_layer_size, $parallelize, $num_seconds" >> speed_tests/trainer.chpl.csv

            # Print elapsed time
            echo "Elapsed time: $num_seconds"
        done
    done
done