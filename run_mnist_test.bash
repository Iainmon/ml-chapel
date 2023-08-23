#!/bin/bash
#SBATCH -t 3-8:0
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=chapdl
#SBATCH --output=job.output

export GASNET_SPAWNFN=S
export GASNET_SSH_SERVERS=`scontrol show hostnames | xargs echo`

cd ml-study/performance && ../../build/mnistCNN > mnistCNNBatch.good