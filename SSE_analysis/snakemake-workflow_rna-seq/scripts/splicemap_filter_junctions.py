import pandas as pd
import argparse
from splicemap import SpliceCountTable

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Filter junctions and export matrix + BED.")
parser.add_argument("--input", required=True, help="Input junction matrix CSV")
parser.add_argument("--output_matrix", required=True, help="Output filtered junction matrix TSV")
parser.add_argument("--output_bed", required=True, help="Output filtered junctions BED file")
args = parser.parse_args()

input_file = args.input
output_file = args.output_matrix
output_bed = args.output_bed

quantile_cutoff = 1       # 90th percentile read count must be > 1
event_median_cutoff = 1   # 5'/3' site median read count must be ≥ 1

# --- Create SpliceCountTable object ---
print("Converting to SpliceCountTable...")
ct = SpliceCountTable.read_csv(input_file)
print(f"Total junctions before filtering: {len(ct.df)}")

# --- Apply 90th percentile filter ---
print(f"\nApplying quantile filter (90th percentile > {quantile_cutoff})...")
ct_quantile = ct.quantile_filter(quantile=90, min_read=quantile_cutoff)
print(f"Junctions after quantile filter: {len(ct_quantile.df)}")

# --- Apply donor (5') and acceptor (3') median filters ---
print(f"\nApplying event5 (donor) median filter (median ≥ {event_median_cutoff})...")
ct_5_quantile = ct_quantile.event5_median_filter(cutoff=event_median_cutoff)
print(f"Junctions after donor filter: {len(ct_5_quantile.df)}")

print(f"\nApplying event3 (acceptor) median filter (median ≥ {event_median_cutoff})...")
ct_3_quantile = ct_quantile.event3_median_filter(cutoff=event_median_cutoff)
print(f"Junctions after acceptor filter: {len(ct_3_quantile.df)}")

# --- Union: keep junctions that passed either 5' or 3' site filter ---
print("\nConcatenating donor and acceptor filtered junctions (union)...")
concat_df = pd.concat([ct_5_quantile.df, ct_3_quantile.df], ignore_index=True)
print(f"Junctions after concatenation (before deduplication): {len(concat_df)}")

# Drop duplicates based on key columns, keeping the first occurrence
key_cols = ["Chromosome", "Start", "End", "Strand"]
deduplicated_df = concat_df.drop_duplicates(subset=key_cols)
deduplicated_df = deduplicated_df.sort_values(by=key_cols).reset_index(drop=True)
print(f"Junctions after dropping duplicates: {len(deduplicated_df)}")

# --- Export filtered matrix ---
print(f"\nSaving filtered junctions to: {output_file}")
deduplicated_df.to_csv(output_file, sep="\t", index=False)

# --- Create and save BED6 file ---
print(f"Saving BED6 file to: {output_bed}")
bed_df = deduplicated_df.copy()
bed_df["chrom"] = bed_df["Chromosome"]
bed_df["start"] = bed_df["Start"]
bed_df["end"] = bed_df["End"]
bed_df["name"] = bed_df["Chromosome"].astype(str) + "_" + bed_df["Start"].astype(str) + "_" + bed_df["End"].astype(str) + "_" + bed_df["Strand"].astype(str)
bed_df["score"] = bed_df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce').sum(axis=1)
bed_df["strand"] = bed_df["Strand"]

# Rearrange BED6 columns
bed_df = bed_df[["chrom", "start", "end", "name", "score", "strand"]]

# Save as BED
bed_df.to_csv(output_bed, sep="\t", header=False, index=False)

print("Done.")
