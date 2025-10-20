#!/bin/sh
#SBATCH --job-name=plot_splice_puffin
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --account=pi-yangili1
#SBATCH -o plot_splice_puffin.log

source /project2/yangili1/yhsz5566/mambaforge/etc/profile.d/conda.sh

conda activate puffin

#python plot_annotated_site_CDF_merged.py -i splice_table_Human.txt -p splice_table_Human.Pangolin.txt -g gencode.v44.annotation.gtf -pg gencode.v34.annotation.gtf  -m MANE_human/release_1.4/MANE.GRCh38.v1.4.ensembl_genomic.gtf  \
#        -o annotated_splice_sites_CDF_merged.pdf > annotated_splice_sites_CDF.summary.txt
# Plot multiple samples with custom threshold
python plot_predictions_vs_targets_DA.py \
    --model_path ./models/model.rep0.pth \
    --test_data ../../create_dataset/Annotated_SSE_Based_datasets/dataset_test_1.h5 \
    --sample_idx 1000 \
    --num_samples 100 \
    --output_dir ./models/plots \
    --gtf ../resources/gencode.v44.annotation.db