#!/bin/bash
#SBATCH --job-name=DataSplit
# number of independent tasks for this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=1:00:00

# From here on, we can start our program
python3 split-code.py
