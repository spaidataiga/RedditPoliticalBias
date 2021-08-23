#!/bin/bash
#SBATCH --job-name=Fix
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=05:00:00
#Number of independent tasks
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=1
# From here on, we can start our program
python3 fix_subreddits.py $1 $2
