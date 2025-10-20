#!/bin/sh
#SBATCH --job-name=train_splice_puffin_reps
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --account=pi-yangili1
#SBATCH -o train_splice_puffin.out
#SBATCH -e train_splice_puffin.err

# ----------------------------
# Environment setup
# ----------------------------
source /project2/yangili1/yhsz5566/mambaforge/etc/profile.d/conda.sh
conda activate puffin

# Create log and result directories if missing
mkdir -p train_parallel

# ----------------------------
# File paths
# ----------------------------
TRAIN_DATA=../../create_dataset/Annotated_SSE_Based_datasets/dataset_train_all.h5
TEST_DATA=../../create_dataset/Annotated_SSE_Based_datasets/dataset_test_1.h5 # need to change to the one without paralogs
SAVE_DIR=train_parallel

# ----------------------------
# Run multiple replicates in parallel
# Adjust this list depending on available VRAM (e.g., 4 replicates for a 24GB GPU)
# ----------------------------

# do not use 11 since it crashes
for RID in 4 5 6 7
do
  echo "Launching replicate $RID..."
  python train_splice_puffin_DA_parallel.py \
      --train_data $TRAIN_DATA \
      --test_data $TEST_DATA \
      --save_dir $SAVE_DIR \
      --replicate_id $RID \
      --epochs 100 \
      > $SAVE_DIR/rep_${RID}.log 2>&1 &
  sleep 10  # Optional: stagger start slightly to avoid GPU spikes
done

wait
echo "All replicates finished successfully."
