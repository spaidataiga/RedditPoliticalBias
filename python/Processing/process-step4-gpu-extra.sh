#!/bin/bash
#SBATCH --job-name=4xProcess
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#ensure only one split is processed at a time
#gpu partition
#SBATCH -p gpu --gres=gpu:titanx:2
#time limit is 6 hours

# From here on, we can start our program
echo $CUDA_VISIBLE_DEVICES
printf -v SPLIT "%02g" ${SLURM_ARRAY_TASK_ID}
python3 step-two-extra.py $1 $2 $SPLIT

#python3 step-twoX.py $1 $2 $3
