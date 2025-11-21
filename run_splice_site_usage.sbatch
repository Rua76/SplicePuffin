#!/bin/bash
#SBATCH --job-name=run_splice_site_usage
#SBATCH --time=8:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --account=pi-xuanyao
#SBATCH -o splice_site_usage.out
#SBATCH -e splice_site_usage.err

set -e  # Exit on error

# Load conda and activate base environment with snakemake installed
source /project2/yangili1/yhsz5566/mambaforge/etc/profile.d/conda.sh
#conda activate rna-seq_simplequantification  # replace with your actual snakemake environment name
conda activate r_env
# Run Snakemake with 4 cores
snakemake -s splice_site_usage_from_bams.smk --cores $SLURM_CPUS_PER_TASK --rerun-incomplete 
snakemake -s splice_site_table.smk --cores $SLURM_CPUS_PER_TASK --rerun-incomplete

echo "Snakemake splicing analysis completed."
