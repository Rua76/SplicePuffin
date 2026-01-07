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
TRAIN_DATA=../create_dataset/dataset_train_all.h5
TEST_DATA=../create_dataset/filtered_test_1.h5
BASE_SAVE_DIR=./experiments
mkdir -p $BASE_SAVE_DIR

# ----------------------------
# Run multiple replicates in parallel
# Adjust this list depending on available VRAM (e.g., 4 replicates for a 24GB GPU)
# ----------------------------
#  7 8 9 10 11 12 
# ----------------------------
# 3 layers model arch_type options: standard, large_kernel, residual 
# ----------------------------
for RID in 1 2 3 4 5 6
do
  echo "Launching replicate $RID..."
  python train_splice_puffin_DA_parallel.py \
      --train_data $TRAIN_DATA \
      --test_data $TEST_DATA \
      --save_dir $BASE_SAVE_DIR \
      --replicate_id $RID \
      --batch_size 16 \
      --learning_rate 5e-4 \
      --num_layers 3 \
      --arch_type standard  \
      --loss_type bce \
      --epochs 500 &
  sleep 10
done

wait
echo "All replicates finished successfully."
