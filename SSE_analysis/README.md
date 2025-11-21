# SSE_analysis

This directory contains the Snakemake workflow and supporting files used to perform **Splice Site Usage (SSE) analysis** from RNA-seq BAM files. 
---

## Directory Contents

### `snakemake-workflow_rna-seq/`
A folder containing Ben's Spliser implementation. Contains code for:
- Interfacing with BAM files and annotation data
- Parsing splice-junction information
- Quantifying splice site usage (SSE)

Majority of the code for SSE analysis is contained under scripts/ folder within this directory.

---

### `r_env.yml`
A mamba environment file that specifies all necessary R packages and dependencies required to run the SSE analysis workflows.

**To create and activate the environment:**
```bash
mamba env create -f r_env.yml
mamba activate r_env
```

---

### `splice_site_usage_from_bams.smk`
The primary Snakemake workflow for running full **splice site usage (SSE) analysis**. It:
- Collects splice-junction counts directly from BAM files
- Calculates splice-site usage metrics
- Produces summary files for downstream analysis

Paths to input BAM files and reference annotations are specified within this file.

---

### `splice_site_table.smk`
A Snakemake workflow for generating **splice site tables** that will be used for model training. 

---

### `run_splice_site_usage.sbatch`
A SLURM batch script used to run the SSE workflow on an HPC cluster.

**To submit to SLURM:**
```bash
sbatch run_splice_site_usage.sbatch
```


---

## Running the Full SSE Pipeline

### 1. Set up the environment
```bash
mamba env create -f r_env.yml
mamba activate r_env
```

### 2. Run on an HPC cluster using SLURM
```bash
sbatch run_splice_site_usage.sbatch
```



