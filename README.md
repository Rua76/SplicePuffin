# SplicePuffin (temp name)

This repository contains code and pretrained models for training and analyzing the Puffin splice model.

## Files

### `train_splice_puffin_stage1.py`
- Training script for the Puffin splice model.  
- Trains **12 replicates**, each on **randomly sampled 40,000 datapoints**.  
- Each datapoint is an **input sequence of 5,000 bp** containing an arbitrary number of splice sites.  
- Each replicate is trained for **50 epochs**.

### `Process_models_new.ipynb`
- Jupyter notebook modified from `Process_models.ipynb` of the original Puffin model.  
- Used to **extract sequence motifs** from trained model replicates.

### `models.zip`
- Archive containing **pretrained model replicates** from the training stage1.

### `motif_12replicates_selected.pdf`
- Repetitive Motifs extracted with 7 out of 12 replicates. Same as the file shared in the slack channel.

---
## To create data files for model training

Run `./create_files.sh`. This will generate `dataset*.h5` files, which are the training and test datasets (requires ~500GB of space). These can be used in the `train` and `evaluate` steps below.

Dependencies:
- `conda create -c bioconda -n create_files_env python=2.7 h5py bedtools` or equivalent

Inputs: 
- `splice_table_Human.txt`
- Reference genomes for each species from [GENCODE](https://www.gencodegenes.org/) and [Ensembl](https://uswest.ensembl.org/index.html)

Outputs: `dataset_train_all.h5` and `dataset_test_1.h5` (human test sequences)

---

**Note:** Ensure all dependencies required by the Puffin model framework are installed before running the training or motif extraction scripts.
