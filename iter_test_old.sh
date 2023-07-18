#!/bin/sh

start=`date +%s`
./build/classifier --epochs=100 --useNewIter=false > /dev/null
end=`date +%s`

runtime=$((end-start))
echo "Execution time:"
echo $runtime