#!/bin/bash -l

#$ -l h_rt=12:00:00 

# Set SCC project
#$ -P cs640g

# Request 16 CPUs
#$ -pe omp 16

# Request 2 GPU 
#$ -l gpus=2

# Specify the minimum GPU compute capability 
#$ -l gpu_c=6.0

# Send an email when job is ended or aborted
#$ -m ea

# Merge error and output files into a single file
#$ -j y


module load python3/3.8.6
module load pytorch/1.7.0
module load torch/2.1
python main.py