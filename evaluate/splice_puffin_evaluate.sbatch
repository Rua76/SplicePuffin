#!/bin/sh
#SBATCH --job-name=evaluate_splice_puffin
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --account=pi-xuanyao
#SBATCH -o eval_splice_puffin.log

source /project2/yangili1/yhsz5566/mambaforge/etc/profile.d/conda.sh

conda activate puffin

#python evaluate_auprc.py \
#    --test_data ../../create_dataset/Annotated_SSE_Based_datasets/dataset_test_1.h5 \
#    --save_dir train_parallel/ \
#    --replicate_id 11 \
#    --batch_size 64

python plot_motif_effect_DA.py \
    --model_dir ../train/train_parallel \
    --n_models 12 \
    --output_dir ./figures/motif_effects \
    --similarity_threshold 0.95 \
    --replicate_select 1 \
    --replicate_min 7
