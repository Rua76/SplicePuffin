#!/bin/sh
#SBATCH --job-name=create_files
#SBATCH --time=4:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --account=pi-xuanyao
#SBATCH -o dataset.create.log

source /project2/yangili1/yhsz5566/mambaforge/etc/profile.d/conda.sh

conda activate puffin

./create_files.sh

# Remove paralogs from datasets
python remove_paralogs_from_h5.py \
    --train_path dataset_train_all.h5 \
    --test_path dataset_test_1.h5 \
    --min_identity 0.8 \
    --min_coverage 0.7
