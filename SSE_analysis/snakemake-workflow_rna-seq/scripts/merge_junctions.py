import os
import sys
import csv
import re
from collections import defaultdict

# Get inputs from snakemake
input_files = snakemake.input.juncs
output_file = snakemake.output[0]

# Extract sample name from file path
def get_sample_name(file_path):
    return os.path.basename(os.path.dirname(os.path.dirname(file_path)))

sample_names = [get_sample_name(f) for f in input_files]

# Valid chromosomes
valid_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

# Store counts per junction
# key: (chrom, start, end, strand), value: dict of sample → count
junction_counts = defaultdict(lambda: defaultdict(int))

for sample, file_path in zip(sample_names, input_files):
    with open(file_path, 'r') as f:
        for line in f:
            cols = line.strip().split()
            if len(cols) < 9:
                continue

            chrom = cols[0]
            if chrom not in valid_chroms:
                continue  # skip unwanted chromosomes

            start = int(cols[1])
            end = int(cols[2])
            strand_code = cols[3]
            unique_reads = int(cols[6])

            strand = {'1': '+', '2': '-'}.get(strand_code, '.')
            if strand == '.':
                continue

            key = (chrom, start, end, strand)
            junction_counts[key][sample] += unique_reads

# Sort keys for reproducible output
sorted_keys = sorted(junction_counts.keys())

# Write output
with open(output_file, 'w', newline='') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['Chromosome', 'Start', 'End', 'Strand'] + sample_names)
    
    for key in sorted_keys:
        row = list(key)
        row += [junction_counts[key].get(sample, 0) for sample in sample_names]
        writer.writerow(row)
