###############################################################################
"""This parser takes as input the .h5 file produced by create_datafile.py and
outputs a .h5 file with datapoints of the form (X, Y), which can be understood
by Keras models."""
###############################################################################

import h5py
import numpy as np
import sys
import time
from utils_masking import *
from constants import *

start_time = time.time()

# --- argument validation ---
if len(sys.argv) < 6:
    print("Usage: python create_dataset.py [train|test|all] [0|1|all] [species] [start] [fnum]")
    sys.exit(1)

assert sys.argv[1] in ['train', 'test', 'all']
assert sys.argv[2] in ['0', '1', 'all']

species = sys.argv[3]
start = int(sys.argv[4])  # 0 if creating new file, 1 if appending to existing file
fnum = int(sys.argv[5])
data_dir = './'

input_path = f"{data_dir}datafile_{species}_{fnum}_{sys.argv[1]}_all.h5"

with h5py.File(input_path, 'r') as h5f:
    print(len(h5f['SEQ']))
    print(len(h5f['TX_START']))

    NAME = h5f['NAME'][:]
    CHROM = h5f['CHROM'][:]
    SEQ = h5f['SEQ'][:]
    STRAND = h5f['STRAND'][:]
    TX_START = h5f['TX_START'][:]
    TX_END = h5f['TX_END'][:]
    JN_START = h5f['JN_START'][:]

print("data loaded")

# --- open output file ---
output_path = f"{data_dir}dataset_{sys.argv[1]}_{sys.argv[2]}.h5"

if start == 0:
    h5f2 = h5py.File(output_path, 'w')
else:
    h5f2 = h5py.File(output_path, 'a')
    assert len(h5f2) % 4 == 0
    start = len(h5f2) // 4

ctr = start

# --- main loop ---
for idx in range(SEQ.shape[0]):
    # decode bytes from HDF5 string datasets (Python3 requirement)
    name = NAME[idx].decode() if isinstance(NAME[idx], (bytes, np.bytes_)) else NAME[idx]
    chrom = CHROM[idx].decode() if isinstance(CHROM[idx], (bytes, np.bytes_)) else CHROM[idx]
    strand = STRAND[idx].decode() if isinstance(STRAND[idx], (bytes, np.bytes_)) else STRAND[idx]
    tx_start = TX_START[idx].decode() if isinstance(TX_START[idx], (bytes, np.bytes_)) else TX_START[idx]
    tx_end = TX_END[idx].decode() if isinstance(TX_END[idx], (bytes, np.bytes_)) else TX_END[idx]
    jn_start = JN_START[idx].decode() if isinstance(JN_START[idx], (bytes, np.bytes_)) else JN_START[idx]
    seq = SEQ[idx].decode() if isinstance(SEQ[idx], (bytes, np.bytes_)) else SEQ[idx]

    # Create data points
    X, A0, A1, mask = create_datapoints(seq, strand, tx_start, tx_end, jn_start)

    for i, _ in enumerate(A0[0]):
        # Skip if no splice sites
        if np.sum(A0[0][i][:, 0]) < 1 or np.sum(A0[0][i][:, 1]) < 1:
            continue

        # --- sequence ---
        h5f2.create_dataset(f'X{ctr}', data=X[i])

        # --- labels (spliced/unspliced) ---
        h5f2.create_dataset(f'Y{ctr}', data=A0[0][i])

        # --- mask (valid positions) ---
        h5f2.create_dataset(f'M{ctr}', data=mask[0][i])

        # --- coordinates ---
        if strand == '+':
            coords = np.array([
                chrom,
                str(int(tx_start) - (CL_max // 2) + SL * i - 1),
                str(int(tx_start) + SL * (i + 1) + CL_max // 2 - 1),
                strand
            ], dtype='S')
        elif strand == '-':
            coords = np.array([
                chrom,
                str(int(tx_end) + (CL_max // 2) - SL * i),
                str(int(tx_end) - SL * i - SL - CL_max // 2),
                strand
            ], dtype='S')
        else:
            print("Error: invalid strand")
            h5f2.close()
            sys.exit(1)

        h5f2.create_dataset(f'Z{ctr}', data=coords)
        ctr += 1

h5f2.close()
print(f"Total records written: {ctr}")
print(f"--- {time.time() - start_time:.2f} seconds ---")
###############################################################################
