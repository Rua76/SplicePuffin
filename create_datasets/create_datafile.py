###############################################################################
"""This parser takes as input the text files canonical_dataset.txt and 
canonical_sequence.txt, and produces a .h5 file datafile_{}_{}.h5,
which will be later processed to create dataset_{}_{}.h5. The file
dataset_{}_{}.h5 will have datapoints of the form (X,Y), and can be
understood by Keras models."""
###############################################################################

import numpy as np
import re
import sys
import time
import h5py
from constants import *

start_time = time.time()

# --- Argument validation ---
if len(sys.argv) < 5:
    print("Usage: python parser.py [train|test|all] [0|1|all] [fnum] [Human|Mouse|...]")
    sys.exit(1)

assert sys.argv[1] in ['train', 'test', 'all']
assert sys.argv[2] in ['0', '1', 'all']  # ortholog status

fnum = int(sys.argv[3])  # split into multiple files to avoid memory issues
splice_table = f"splice_table_{sys.argv[4]}.txt"
sequence = f"canonical_sequence_{sys.argv[4]}.txt"
data_dir = './'

# --- Chromosome groups ---
if sys.argv[1] == 'train':
    CHROM_GROUP = [
        'chr2', 'chr4', 'chr6', 'chr8', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY'
    ]
elif sys.argv[1] == 'test':
    CHROM_GROUP = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
else:
    CHROM_GROUP = [
        'chr2', 'chr4', 'chr6', 'chr8', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY',
        'chr1', 'chr3', 'chr5', 'chr7', 'chr9'
    ]

###############################################################################

NAME = []      # Gene symbol
PARALOG = []   # 0 if no paralogs exist, 1 otherwise
CHROM = []     # Chromosome number
STRAND = []    # Strand in which the gene lies (+ or -)
TX_START = []  # Position where transcription starts
TX_END = []    # Position where transcription ends
JN_START = []  # Positions where canonical exons end
SEQ = []       # Nucleotide sequence

ctr = 0

with open(splice_table, 'r') as fpr1, open(sequence, 'r') as fpr2:
    for line1 in fpr1:
        line2 = fpr2.readline()
        if not line2:
            break  # handle uneven file lengths safely

        data1 = re.split(r'\n|\t', line1.strip())
        data2 = re.split(r'\n|\t|:|-', line2.strip())

        # Safety check
        if len(data1) < 7 or len(data2) < 4:
            continue

        assert data1[2] == data2[0]
        assert int(data1[4]) == int(data2[1]) + CL_max // 2 + 1
        assert int(data1[5]) == int(data2[2]) - CL_max // 2

        if ("Human" in sys.argv[4] and data1[2] not in CHROM_GROUP):
            continue

        if (sys.argv[2] != data1[1]) and (sys.argv[2] != 'all'):
            continue

        ctr += 1
        if ctr < (fnum - 1) * 5000 or ctr >= fnum * 5000:
            continue

        NAME.append(data1[0])
        PARALOG.append(int(data1[1]))
        CHROM.append(data1[2])
        STRAND.append(data1[3])
        TX_START.append(data1[4])
        TX_END.append(data1[5])
        JN_START.append(data1[6])
        SEQ.append(data2[3])

###############################################################################

output_filename = (
    f"{data_dir}datafile_{sys.argv[4]}_{fnum}_{sys.argv[1]}_{sys.argv[2]}.h5"
)

with h5py.File(output_filename, 'w') as h5f:
    h5f.create_dataset('NAME', data=np.asarray(NAME, dtype='S'))
    h5f.create_dataset('PARALOG', data=np.asarray(PARALOG))
    h5f.create_dataset('CHROM', data=np.asarray(CHROM, dtype='S'))
    h5f.create_dataset('STRAND', data=np.asarray(STRAND, dtype='S'))
    h5f.create_dataset('TX_START', data=np.asarray(TX_START, dtype='S'))
    h5f.create_dataset('TX_END', data=np.asarray(TX_END, dtype='S'))
    h5f.create_dataset('JN_START', data=np.asarray(JN_START, dtype='S'))
    h5f.create_dataset('SEQ', data=np.asarray(SEQ, dtype='S'))

print(f"--- {time.time() - start_time:.2f} seconds ---")
###############################################################################
