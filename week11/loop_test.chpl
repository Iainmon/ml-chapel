

import Random;
import Time;
import Math;

config const n = 100000;
config const mode = "serial";

var arr: [0..#n] real;

Random.fillRandom(arr);


var tic = new Time.stopwatch();
tic.start();

if mode == "serial" {
    for i in arr.domain {
        arr[i] = Math.exp(arr[i]);
    }
}

if mode == "parallel" {
    forall i in arr.domain {
        arr[i] = Math.exp(arr[i]);
    }
}

if mode == "vectorized" {
    foreach i in arr.domain {
        arr[i] = Math.exp(arr[i]);
    }
}

writeln("Time: ", tic.elapsed(), " s");

var sum = + reduce arr;

writeln("Sum: ", sum);
