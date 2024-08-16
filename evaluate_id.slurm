#!/bin/bash -l

# # Resource request
#SBATCH -p gpu-mxian
#SBATCH --gpus 1
#SBATCH --nodelist=node01
#SBATCH -o evaluate_id.out

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# for dataset in SVHN CIFAR10; do
for dataset in SVHN; do

  # LULAs
  python evaluate_id.py --dataset $dataset --base plain > "$dataset".out
	
done
