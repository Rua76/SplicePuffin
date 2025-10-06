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

**Note:** Ensure all dependencies required by the Puffin model framework are installed before running the training or motif extraction scripts.
