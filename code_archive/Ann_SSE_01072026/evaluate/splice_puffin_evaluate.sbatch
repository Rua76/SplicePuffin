#!/bin/sh
#SBATCH --job-name=evaluate_splice_puffin
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --account=pi-yangili1
#SBATCH -o connor_test.log

source /project2/yangili1/yhsz5566/mambaforge/etc/profile.d/conda.sh

conda activate pangolin
module load cuda/11.7
python evaluate_AUPRC.py \
    --test_data ../../create_dataset/Annotated_datasets/filtered_test_test1.h5 \
    --save_dir twolayers_KL_beta2_only_models/replicate_6/ \
    --mode with_ref_genome \
    --reference_file ../resources/GRCh38.primary_assembly.genome.fa \
    --num_replicates 6 \
    --skip_spliceai \
    --skip_maxent \
    --output_file twolayers_KL_beta2_only_auprc.pdf \
    --batch_size 128

# Run with multiple models
python evaluate_connor_testset_scatter.py \
  --data ../../create_dataset/Connor_testest/Hao_test_set_w_Maxent_scores_20251017_.tsv \
  --out_dir Connor_Exon_testset_results/scatter_plots \
  --reference ../resources/GRCh38.primary_assembly.genome.fa \
  --models simple_binary:trained_models_2025Dec9/train_parallel_Binary/model.rep6.pth:simple_binary \
        twolayers_BCE:trained_models_2025Dec9/twolayers_BCE_SSE_models/replicate_6/model.rep6.pth:2layer \
        twolayers_KL:trained_models_2025Dec9/twolayers_KL_SSE_models/replicate_6/model.rep6.pth:2layer \
        twolayers_BCE_beta2_only:trained_models_2025Dec9/twolayers_BCE_beta2_only_models/replicate_6/model.rep6.pth:2layer \
        twolayers_KL_beta2_only:trained_models_2025Dec9/twolayers_KL_beta2_only_models/replicate_6/model.rep6.pth:2layer \
        triple_layers_factorized_BCE:trained_models_2025Dec9/train_parallel_SSE_triple_layers_factorized_BCE/replicate_6/model.rep6.pth:3layer \
        triple_layers_factorized_KL:trained_models_2025Dec9/train_parallel_SSE_triple_layers_factorized_KL/replicate_6/model.rep6.pth:3layer \
        spliceai:../../SpliceAI/spliceai/models/spliceai1.h5:spliceai

conda activate puffin
python plot_motif_effect_DA_SSE.py