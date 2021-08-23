#!/bin/bash
#SBATCH --job-name=3xProcess
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#ensure only one split is processed at a time
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#Skipping many options! see man sbatch
# From here on, we can start our program

printf -v SPLIT "%04g" ${SLURM_ARRAY_TASK_ID}
python3 step-one-3x.py $1 $2 $SPLIT

#python3 step-one-3x.py $1 $2 ${SLURM_ARRAY_TASK_ID}

#ENVIRONMENT 1
