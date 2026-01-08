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
    Current model is `SimpleNet_TwoLayers` and `SimpleNet_TripleLayers` which predicts both donor and acceptor splice sites simultaneously:
    
    ```python  
    class SimpleNet_TwoLayers(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, 40, kernel_size=51, padding=25)
        self.activation = nn.Softplus()
        self.deconv = FFTConv1d(40, 2, kernel_size=601, padding=300)

    def forward(self, x):
        y = self.activation(self.conv(x))
        y_pred = torch.sigmoid(self.deconv(y))  # independent sigmoid per channel
        return y_pred[:, :, 500:-500]
    ```

    ```python
    class SimpleNet_TripleLayers(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.motif_layer = nn.Conv1d(input_channels, 40, kernel_size=51, padding=25)
        #self.synergy_layer = FFTConv1d(40, 40, kernel_size=401, padding=200)
        self.synergy_layer = FactorizedFFTConvBlock(
            in_channels=40,
            out_channels=40,
            mid_channels=4,   # your bottleneck
            kernel_size=401,
            padding=200
        )

        self.effect_layer = FFTConv1d(40, 2, kernel_size=601, padding=300)
        self.softplus = nn.Softplus()

    def forward(self, x):
        y = self.softplus(self.motif_layer(x))
        motifact = torch.sigmoid(self.synergy_layer(y)) * y
        y_pred = torch.sigmoid(self.effect_layer(motifact))  # independent sigmoid per channel
        return y_pred[:, :, 500:-500]
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

**Warning** Code NOT tested in fresh installed environment! 

**Warning** Highly recommend not running tests with SpliceAI as current build do not include the code for Installing Tensorflow, which is a DISASTER to do! I can share SpliceAI prediction if you are interested.

### *OUTDATED* `evaluate_auprc.py` 
calculate the AUPRC value of a specific replicate. Use the following command to evaluate a specific replicate:

    ```bash
    python evaluate_auprc.py \
        --test_data path/to/dataset_test.h5 \
        --save_dir path/to/trained_models \
        --replicate_id <replicate_number> \
        --batch_size 32 \
        --num_workers 4
    ```

### `evaluate_connor_testset_scatter.py`

This script evaluates multiple splice site prediction models on an exon-based test dataset and generates **AUPRC scatter plots** for direct comparison across models. It supports SimpleNet variants, binary classifiers, and SpliceAI models, with automatic or explicit model type detection.

---

#### Overview

- Runs inference for multiple splice site prediction models on a shared exon test set
- Computes AUPRC metrics for each model
- Produces scatter plots comparing model performance
- Supports caching of predictions to speed up repeated runs
- Allows flexible specification of heterogeneous model types

---

#### Required arguments

- `--data` (**required**): Path to the exon dataset in TSV format.
- `--out_dir` (**required**): Output directory for prediction files, metrics, and figures.

---

#### Optional arguments

- `--reference`: Path to reference genome FASTA file.  
  Required when evaluating SpliceAI models.
- `--force_rerun`: Force recomputation of model predictions even if cached `.npz` files already exist.

---

#### Model specification (`--models`)

Models are specified using the `--models` argument, which may be provided multiple times. Each model is defined using one of the following formats:
    - tag:path
    - tag:path:type


Where:
- `tag` is a short label used in plots and output filenames
- `path` is the path to the model checkpoint or model directory
- `type` specifies the model architecture or inference backend

If `type` is omitted or set to `auto`, the script attempts to automatically detect the model type.

---

#### Supported model types

The following model type strings are recognized (case-insensitive):

- `2layer`, `simple_2layer` → `simple_2layer`
- `3layer`, `simple_3layer`, `triple` → `simple_3layer`
- `binary`, `simple_binary` → `simple_binary`
- `spliceai` → `spliceai`
- `auto` → automatic model type detection

Unknown or unsupported type strings fall back to automatic detection with a warning.

---

#### Model parsing behavior

- Model specifications are parsed at runtime and normalized into a unified internal format.
- If no models are provided, the script exits with a usage warning and example commands.
- All valid models are listed before evaluation begins for transparency.

---

#### Example usage

```bash
python evaluate_connor_testset_scatter.py \
  --data ../../create_dataset/Connor_testest/Hao_test_set_w_Maxent_scores_20251017_.tsv \
  --out_dir Connor_Exon_testset_results/scatter_plots \
  --reference ../resources/GRCh38.primary_assembly.genome.fa \
  --models simple_binary:trained_models_2025Dec9/train_parallel_Binary/model.rep6.pth:simple_binary \
        twolayers_BCE:trained_models_2025Dec9/twolayers_BCE_SSE_models/replicate_6/model.rep6.pth:2layer \
        triple_layers_factorized_BCE:trained_models_2025Dec9/train_parallel_SSE_triple_layers_factorized_BCE/replicate_6/model.rep6.pth:3layer \
        spliceai:../../SpliceAI/spliceai/models/spliceai1.h5:spliceai

```

#### Outputs

- Cached model predictions (.npz)
- AUPRC metrics for each model
- Scatter plot figures comparing model performance
- Console summary of evaluated models

---

### `simpleModel_InteractiveEffectCurves.ipynb`

Model kernel live visualization and motif extraction using interactive effect curves.

Just drop the trained model file (e.g., `model.rep7.pth`) into the same directory as the notebook and run all cells to visualize the learned motifs and effect curves.

Only support two-layer SimpleNet models for now.

**Note:** Ensure all dependencies required by the Puffin model framework are installed before running the training or motif extraction scripts.
