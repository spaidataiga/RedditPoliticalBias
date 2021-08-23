#!/bin/bash
#SBATCH --job-name=Scraper
# number of independent tasks for this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4

# From here on, we can start our program
python3 cmt_scraper.py
