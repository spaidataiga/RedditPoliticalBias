#!/bin/bash
#SBATCH --job-name=trainSVM
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#gpu partition
#SBATCH -p gpu --gres=gpu:titanx:2
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#Skipping many options! see man sbatch
# From here on, we can start our program
python3 train_model2.py $1
