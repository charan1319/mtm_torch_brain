#!/bin/bash
#SBATCH --job-name=ndt2_train      # Job name
#SBATCH --output=logs/train_%j.log # Standard output and error log
#SBATCH --nodes=1                  # Run all processes on a single node	
#SBATCH --ntasks=1                 # Run a single task		
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=32G                  # Job memory request
#SBATCH --time=6:00:00           # Time limit hrs:min:sec
#SBATCH --partition=gpuA100x8      # Partition/Queue name
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --account=bcxj-delta-gpu   # Project allocation account

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Number of GPUs: $SLURM_NTASKS"

# Activate your conda environment
source activate torchbrain

# Navigate to your project directory
cd $SLURM_SUBMIT_DIR

# Run the training script
python train.py

# Print completion time
echo "End time: $(date)" 