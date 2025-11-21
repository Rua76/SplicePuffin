import pandas as pd
import numpy as np

# Inputs and outputs from Snakemake
input_file = snakemake.input.raw_counts_bed
output_file = snakemake.output.normalized_tsv

# Read the BED + raw count matrix with header
bed_df = pd.read_csv(input_file, sep="\t", comment=None)

# Strip leading '#' from first column name if present
bed_df.columns = [col.lstrip('#') for col in bed_df.columns]

# First 4 columns are metadata (BED)
metadata_cols = bed_df.columns[:4]
count_cols = bed_df.columns[4:]

# Convert count columns to numeric
counts = bed_df[count_cols].apply(pd.to_numeric, errors="coerce")

print("\n[DEBUG] Raw counts summary:")
print(counts.describe())

# Step 1: Normalize to CPM (Counts Per Million) per sample
cpm = counts.divide(counts.sum(axis=0), axis=1) * 1_000_000

print("\n[DEBUG] CPM normalized counts summary (first few samples):")
print(cpm.iloc[:, :5].describe())  # show first 5 samples

# Step 2: Tissue-specific preprocessing
tissues = ["Heart", "Brain", "Liver", "Testis"]

processed = {}
for tissue in tissues:
    # Select all columns (samples) belonging to this tissue
    tissue_cols = [c for c in count_cols if tissue in c]
    if not tissue_cols:
        continue

    tissue_df = cpm[tissue_cols]

    # 1. Clip at 99.99th percentile (computed across all values in this tissue)
    clip_val = np.percentile(tissue_df.values, 99.99)
    clipped = tissue_df.clip(upper=clip_val)

    print(f"\n[DEBUG] {tissue} processing:")
    print(f"  Number of samples: {len(tissue_cols)}")
    print(f"  Global 99.99th percentile cutoff: {clip_val:.2f}")
    print("  Example sample before/after clipping:")
    example_col = tissue_cols[0]
    print("    Before:", tissue_df[example_col].describe())
    print("    After :", clipped[example_col].describe())

    # 2. Scale by mean of active junctions (>0 after clipping), per sample
    scaled = clipped.copy()
    for col in tissue_cols:
        active = clipped[col] > 0
        mean_active = clipped.loc[active, col].mean()
        if pd.notna(mean_active) and mean_active > 0:
            scaled[col] = clipped[col] / mean_active
        else:
            scaled[col] = 0.0  # if no active junctions, set to 0

        # Debug per sample
        print(f"    {col}: mean_active={mean_active:.4f}, "
              f"min_scaled={scaled[col].min():.4f}, max_scaled={scaled[col].max():.4f}")

    processed.update({col: scaled[col] for col in tissue_cols})

# Reassemble full DataFrame with metadata + normalized values
processed_df = pd.DataFrame(processed)
output_df = pd.concat([bed_df[metadata_cols], processed_df], axis=1)

# Save to TSV
output_df.to_csv(output_file, sep="\t", index=False)

print("\n[DEBUG] Final processed data (first 5 rows):")
print(output_df.head())
