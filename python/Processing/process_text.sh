#!/bin/bash
#SBATCH --job-name=DataSplit
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#ensure only one split is processed at a time
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#Skipping many options! see man sbatch
# From here on, we can start our program

python3 text_process.py $1 $2

#python3 step-one-3x.py $1 $2 ${SLURM_ARRAY_TASK_ID}
