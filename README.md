# SplicePuffin (temp name)

This repository contains code and pretrained models for training and analyzing the Puffin splice model.

---
## To create data files for model training (~2 hours)

1. use mamba or conda to create a new environment (python version 3.11):
    
    ```bash
    mamba create -n puffin python=3.11 -y
    mamba activate puffin
    ```

    Install pytorch with CUDA 11.8
    ```bash
    pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    Install `bedtools` with mamba/conda, this cannot be done with pip
    ```bash
    mamba install -c bioconda bedtools
    ```

    Install the rest from `requirements.txt`
    ```bash
    pip install -r requirements.txt
    ```
3. Prepare your input data files (FASTA and splice table file) and place them in the `resources` directory.

    Create a `resources` directory if it does not exist:
    ```bash
    mkdir resources
    cd resources
    ```

    Get `GRCh38.primary_assembly.genome.fa`:
    ```bash
    wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/GRCh38.primary_assembly.genome.fa.gz
    # unzip
    gunzip GRCh38.primary_assembly.genome.fa.gz
    ```
    Get `gencode.v44.annotation.gtf`:
    ```bash
    wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
    # unzip
    gunzip gencode.v44.annotation.gtf.gz
    ```

4. The table of splice sites is needed. An example row looks like this:

    ```
    ENSG00000278267.1	0	chr1	-	17368	17436	chr1|17368|-|A:0.17154942528735637;
    ```

    Here, the columns represent:
    - Gene ID
    - 0 (used to indicate paralog, now deprecated, just set to 0)
    - Chromosome
    - Strand
    - Start position
    - End position
    - The coordinates, type (Donor or acceptor) and splice site strength estimate (SSE) value of a splice site. ';' is used to separate multiple splice sites for a gene.

    To train your own model, you need to create a similar table for your dataset.

5. Navigate to the `create_datasets` directory and run the `create_dataset_files.sbatch` script to generate the dataset files:

    ```bash
    sbatch create_dataset_files.sbatch
    ```
    
    This will create the necessary HDF5 files for training and testing, removing paralogous sequences as specified in the script.
   
    - Training dataset file: `dataset_train_all.h5`
    - Testing dataset (paralogs removed) file: `filtered_test_1.h5`

---

## To train the model (12 hours)

All codes are stored in the `train` directory.

1. Model architectures are stored in the `train/models.py` file. You can modify the architecture as needed.
    Current model is `SimpleNetModified_DA`, which predicts both donor and acceptor splice sites simultaneously:
    
    ```python  
    class SimpleNetModified_DA(nn.Module):
        def __init__(self, input_channels=4):
            super(SimpleNetModified_DA, self).__init__()
            self.conv = nn.Conv1d(input_channels, 40, kernel_size=51, padding=25)
            self.activation = nn.Softplus()

            # Separate deconv layers for donor and acceptor 
            self.deconv_donor = FFTConv1d(40, 1, kernel_size=601, padding=300) 
            self.deconv_acceptor = FFTConv1d(40, 1, kernel_size=601, padding=300)  

        def forward(self, x):
            y = self.conv(x)  # Shape: (batch_size, 40, 5000)
            # activation
            yact = self.activation(y)   # Shape: (batch_size, 40, 5000)
            # Separate predictions for donor and acceptor
            y_pred_donor = torch.sigmoid(self.deconv_donor(yact))  # Shape: (batch_size, 1, 5000)
            y_pred_acceptor = torch.sigmoid(self.deconv_acceptor(yact))  # Shape: (batch_size, 1, 5000)
            
            # Concatenate the outputs along channel dimension
            y_pred = torch.cat([y_pred_donor, y_pred_acceptor], dim=1)  # Shape: (batch_size, 2, 5000)
            
            # Crop the output to match label shape: 5000 -> 4000 by removing 500 from each side
            return y_pred[:, :, 500:-500]  # Final shape: (batch_size, 2, 4000)
    ```

2. Navigate to the `splice_puffin_parallel_train.sbatch` script and adjust the file paths for training and testing datasets as needed.

3. The number of models trained on one GPU is set to 6 by default. Before submitting the job, you need to manually change the `RID` variable in the `splice_puffin_parallel_train.sbatch` script to a unique identifier for your training run.
   
4. Submit the training job using SLURM:

    ```bash
    sbatch splice_puffin_parallel_train.sbatch
    ```

    There will be 12 models replicates trained. loss will be logged along the training process. MSE, R^2 and Pearson correlation will be logged out every 100 epochs (Note: computation might be inapporpriate, just for monitoring the training process!) The trained models will be saved in `BASE_SAVE_DIR`.

---

### Model training script

`train_splice_puffin_DA_parallel.py`

This script trains a **SimpleNet** convolutional neural network for genomic sequence prediction using HDF5-formatted datasets. It supports multiple model architectures, configurable depth, different loss functions, and parallel replicate training.

#### Overview

- Trains a SimpleNet model on a training H5 dataset
- Evaluates performance on a separate test H5 dataset
- Supports 2-layer and 3-layer architectures with multiple variants
- Allows parallel training via replicate IDs
- Saves trained models and logs to a specified directory

---

#### Required arguments

- `--train_data` (**required**): Path to the training dataset in HDF5 (`.h5`) format.
- `--test_data` (**required**): Path to the test dataset in HDF5 (`.h5`) format.

---

#### Training parameters

- `--batch_size`: Batch size used during training (default: `64`).
- `--learning_rate`: Learning rate for the optimizer (default: `1e-4`).
- `--weight_decay`: Weight decay applied during optimization (default: `0.01`).
- `--validation_split`: Fraction of the training dataset reserved for validation (default: `0.1`).
- `--epochs`: Number of training epochs (default: `100`).

---

#### Model identification and saving

- `--model_name`: Base name used when saving the trained model (default: `simplenet_trained`).
- `--model_num`: Model number or tag for logging and bookkeeping (default: `001`).
- `--save_dir`: Directory where trained models and logs are saved (default: `./models`).
- `--replicate_id`: Replicate ID for parallel or repeated training runs (default: `0`).

---

#### System and runtime options

- `--num_workers`: Number of worker processes used for data loading (default: `4`).
- `--test_mode`: If set, runs in test mode using a reduced subset of data (useful for debugging).

---

#### Model architecture options

- `--num_layers`: Number of layers in the SimpleNet model.
  - `2`: Two-layer SimpleNet
  - `3`: Three-layer SimpleNet  
  (default: `2`)

- `--arch_type`: Architecture variant to use.
  - `standard`: Default SimpleNet architecture
  - `large_kernel`: Three-layer model with larger convolutional kernels
  - `residual`: Three-layer model with residual connections
  - `triple_layers`: Alias for the standard three-layer architecture
  - `softplus`: Three-layer model using softplus activations  
  (default: `standard`)

> Note: Some architecture types are only available for 3-layer models. If an unsupported combination is specified, the script falls back to the standard architecture with a warning.

---

#### Loss function

- `--loss_type`: Loss function used during training.
  - `bce`: Binary cross-entropy loss
  - `kl`: KL-divergence–based loss  
  (default: `bce`)

---

#### Model selection logic

The model architecture is selected internally based on `--num_layers` and `--arch_type`. Unsupported combinations automatically fall back to a standard architecture to ensure training can proceed.

---

#### Typical usage example

```bash
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
      --epochs 500
```

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
