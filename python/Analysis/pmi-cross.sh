#!/bin/bash
#SBATCH --job-name=PMIcross
#Number of independent tasks
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=1
# From here on, we can start our program
python3 pmi-cross.py
