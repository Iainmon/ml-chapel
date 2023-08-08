#!/bin/bash
#SBATCH -t 3-8:0
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=chapel
#SBATCH --output=job.output

export GASNET_SPAWNFN=S
export GASNET_SSH_SERVERS=`scontrol show hostnames | xargs echo`

./build/animalCNN