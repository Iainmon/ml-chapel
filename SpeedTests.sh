#!/bin/bash

rm -rf speed_tests; mkdir speed_tests; make tests fast=true


cp build/week5_classifier speed_tests/week5_classifier
cp build/week6_classifier speed_tests/week6_classifier

echo "[Week5]: (numImages, seconds)"
for num_images in 10 50 100 200 500 1000 2000 5000 10000 20000
do
    start=`date +%s`
    ./speed_tests/week5_classifier --numImages=${num_images} --epochs=1000 > /dev/null
    end=`date +%s`

    runtime=$((end-start))
    echo "$num_images, $runtime"
done


echo "[Week6]: (numImages, seconds)"
for num_images in 10 50 100 200 500 1000 2000 5000 10000 20000
do
    start=`date +%s`
    ./speed_tests/week6_classifier --numImages=${num_images} --epochs=1000 > /dev/null
    end=`date +%s`

    runtime=$((end-start))
    echo "$num_images, $runtime"
done