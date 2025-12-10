#!/bin/sh
#SBATCH --job-name=train_splice_puffin_reps
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --account=pi-yangili1
#SBATCH -o train_splice_puffin.out
#SBATCH -e train_splice_puffin.err

# ----------------------------
# Environment setup
# ----------------------------
source /project2/yangili1/yhsz5566/mambaforge/etc/profile.d/conda.sh
conda activate puffin
# ----------------------------
# File paths
# ----------------------------
TRAIN_DATA=../../create_dataset/Annotated_beta2_only_SSE_datasets/dataset_train_all.h5
TEST_DATA=../../create_dataset/Annotated_beta2_only_SSE_datasets/filtered_test_1.h5
SAVE_DIR=twolayers_KL_beta2_only_models/

# Create log and result directories if missing
mkdir -p $SAVE_DIR
# ----------------------------
# Run multiple replicates in parallel
# Adjust this list depending on available VRAM (e.g., 4 replicates for a 24GB GPU)
# ----------------------------
# do not use 11 since it crashes
#  7 8 9 10 12 13 
for RID in 1 2 3 4 5 6
do
  echo "Launching replicate $RID..."
  python train_splice_puffin_DA_parallel.py \
      --train_data $TRAIN_DATA \
      --test_data $TEST_DATA \
      --save_dir $SAVE_DIR \
      --replicate_id $RID \
      --batch_size 16 \
      --learning_rate 5e-4 \
      --epochs 500  &
  sleep 10  # Optional: stagger start slightly to avoid GPU spikes
done

wait
echo "All replicates finished successfully."
