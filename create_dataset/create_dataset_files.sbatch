#!/bin/sh
#SBATCH --job-name=create_files
#SBATCH --time=2:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --account=pi-xuanyao
#SBATCH -o dataset.create.log

source /project2/yangili1/yhsz5566/mambaforge/etc/profile.d/conda.sh

conda activate SpliceAI

./create_files.sh