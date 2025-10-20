# SplicePuffin (temp name)

This repository contains code and pretrained models for training and analyzing the Puffin splice model.

---
## To create data files for model training

1. use mamba or conda to create a new environment with the required dependencies from 'puffin.yml':
    
    ```bash
    mamba env create -f puffin.yml
    mamba activate puffin
    ```

2. Prepare your input data files (FASTA and splice table file) and place them in the `create_datasets` directory.

    Link to `GRCh38.primary_assembly.genome.fa`:
    ```bash
        https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/GRCh38.primary_assembly.genome.fa.gz
    ```
    Link to `gencode.v44.annotation.gtf`:
    ```bash
        https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
    ```
    Run `creat_db.py` to process the GTF file into a `gffutils` database file:
    ```bash
    python create_db.py --annotation_file gencode.v44.annotation.gtf 
    ```

3. Navigate to the `create_datasets` directory and run the `create_dataset_files.sbatch` script to generate the dataset files:

    ```bash
    sbatch create_dataset_files.sbatch
    ```
    This will create the necessary HDF5 files for training and testing, removing paralogous sequences as specified in the script.

---


---
## To train the model

All codes are stored in the `train` directory.

1. Model architectures are stored in the `train/models.py` file. You can modify the architecture as needed.

2. Navigate to the `splice_puffin_parallel_train.sbatch` script and adjust the file paths for training and testing datasets as needed.

3. The number of models trained on one GPU is set to 4 by default. Before submitting the job, you need to manually change the `RID` variable in the `splice_puffin_parallel_train.sbatch` script to a unique identifier for your training run. Do not use `11` as it result in failed training.

4. Submit the training job using SLURM:

    ```bash
    sbatch splice_puffin_parallel_train.sbatch
    ```

    There will be 12 models replicates trained. AUPRC and loss will be logged along the training process. The trained models and logs will be saved in the specified `SAVE_DIR`.
---

---
## To evaluate the model

All codes are stored in the `evaluate` directory.


1. `evaluate_auprc.py` script is used to calculate the AUPRC value of a specific replicate. Use the following command to evaluate a specific replicate:

    ```bash
    python evaluate_auprc.py \
        --test_data path/to/dataset_test.h5 \
        --save_dir path/to/trained_models \
        --replicate_id <replicate_number> \
        --batch_size 32 \
        --num_workers 4
    ```

2. `plot_predictions_vs_targets_DA.py` script is used to plot the model predictions against the true targets. Use the following command:

    ```bash
    python plot_predictions_vs_targets_DA.py \
        --model_path ./models/model.rep0.pth \
        --test_data ../../create_dataset/Annotated_SSE_Based_datasets/dataset_test_1.h5 \
        --sample_idx 1000 \  # Index of the starting sample to plot
        --num_samples 100 \  # Number of samples to plot
        --output_dir ./models/plots \
        --gtf ../resources/gencode.v44.annotation.db
    ```

    Note that the gtf needs to be processed into a database file using `gffutils` before use.

3. `plot_motif_effect_DA.py` script is used to visualize the motifs and the motif effects learned by the model. Use the following command:

    ```bash
    python plot_motif_effect_DA.py \
        --model_dir ../train/train_parallel \
        --n_models 12 \
        --output_dir ./figures/motif_effects \
        --similarity_threshold 0.95 \
        --replicate_select 1 \
        --replicate_min 7
    ```
---

**Note:** Ensure all dependencies required by the Puffin model framework are installed before running the training or motif extraction scripts.
