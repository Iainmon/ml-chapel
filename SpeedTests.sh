#!/bin/bash

rm -rf speed_tests; mkdir speed_tests; make tests fast=true


cp build/week5_classifier speed_tests/week5_classifier
cp build/week6_classifier speed_tests/week6_classifier

echo "[Python]: (numImages, seconds)"
for num_images in 100 500 1000 2000 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000
do
    start=`date +%s`
    # python3 week2/classifier.py ${num_images} &> /dev/null
    end=`date +%s`

    runtime=$((end-start))
    echo "$num_images, $runtime"
done

echo "[Week5]: (numImages, seconds)"
for num_images in 100 500 1000 2000 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000
do
    start=`date +%s`
    ./speed_tests/week5_classifier --numImages=${num_images} --epochs=1000 > /dev/null
    end=`date +%s`

    runtime=$((end-start))
    echo "$num_images, $runtime"
done


echo "[Week6]: (numImages, seconds)"
for num_images in 100 500 1000 2000 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000
do
    start=`date +%s`
    ./speed_tests/week6_classifier --numImages=${num_images} --epochs=1000 > /dev/null
    end=`date +%s`

    runtime=$((end-start))
    echo "$num_images, $runtime"
done

