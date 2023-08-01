#!/bin/sh

cd /Users/iainmoncrief/Documents/Sandbox
cp build/runCNN bin/runCNN &> /dev/null || echo 1 > /dev/null
/Users/iainmoncrief/Documents/Sandbox/bin/runCNN --modelFile=models/cnn/epoch_1_mnist.cnn.model --normalize=false --recenter=true > /Users/iainmoncrief/Documents/Sandbox/result.txt
