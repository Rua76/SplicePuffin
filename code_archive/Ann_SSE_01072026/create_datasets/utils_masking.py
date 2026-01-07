###############################################################################
"""Functions that process information in .h5 files (datafile_{}_{}.h5)
and convert them into a format usable by Keras."""
###############################################################################

import numpy as np
import re
from math import ceil
from constants import *

assert CL_max % 2 == 0

# One-hot encoding of the inputs:
# 0 is for padding, and 1,2,3,4 correspond to A,C,G,T respectively.
IN_MAP = np.asarray([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

OUT_MAP = np.asarray([
    [0, 0],  # 0: no splicing
    [1, 0],  # 1: donor
    [0, 1],  # 2: acceptor
])


def ceil_div(x, y):
    """Ceiling division, returns ⌈x/y⌉ as int."""
    return int(ceil(float(x) / y))


def reformat_data_X(X0, Y0):
    """Split X0 into overlapping blocks corresponding to output Y0 blocks."""
    assert len(X0) == len(Y0[0]) + CL_max
    num_points = ceil_div(len(Y0[0]), SL)

    Xd = np.zeros((num_points, SL + CL_max))
    X0 = np.pad(X0, [0, SL], 'constant', constant_values=0)

    for i in range(num_points):
        Xd[i] = X0[SL * i:CL_max + SL * (i + 1)]

    return Xd


def reformat_data_Y(Y0):
    """Pad and split Y0 into smaller blocks."""
    num_points = ceil_div(len(Y0[0]), SL)
    Yd = [-np.ones((num_points, SL)) for _ in range(1)]

    Y0 = [np.pad(Y0[t], [0, SL], 'constant', constant_values=-1) for t in range(1)]

    for t in range(1):
        for i in range(num_points):
            Yd[t][i] = Y0[t][SL * i:SL * (i + 1)]

    return Yd


def one_hot_encode_X(Xd):
    """Convert integer-coded sequence to one-hot encoding."""
    return IN_MAP[Xd.astype(np.int8)]


def one_hot_encode_Y(Yd):
    """Convert Y labels to one-hot, replacing negatives with 0."""
    Yd_clean = [np.where(Yd[t] >= 0, Yd[t], 0).astype(np.int8) for t in range(1)]
    return [OUT_MAP[Yd_clean[t]] for t in range(1)]


def create_mask(Yd):
    """Create mask array: 1 for valid positions, 0 for padding."""
    return [np.where(Yd[t] >= 0, 1, 0) for t in range(1)]


# -------------------------------------------------------------------------
# Label binning and usage helpers
# -------------------------------------------------------------------------

def get_usage(cov):
    """Return normalized coverage usage category."""
    if cov == "":
        return 0
    cov = float(cov)
    if cov < -2.99:       # -3
        return -1
    elif cov < -1.99 or cov == 0.0:  # -2 or 0
        return 0
    elif -1.01 < cov < -0.99:        # -1
        return -1
    else:
        return cov


def get_bin(cov, strand=None, site_type=None):
    """Return numeric bin index for coverage & site type."""
    if cov == "":
        return 0
    cov = float(cov)
    if cov < -2.99:
        return 5  # no expression
    elif cov < -1.99:
        return 0  # no splicing
    else:
        if strand is not None and site_type is not None:
            if site_type == 'A':
                return 2  # acceptor
            elif site_type == 'D':
                return 1  # donor
        return 0


# -------------------------------------------------------------------------
# Core datapoint creation
# -------------------------------------------------------------------------

def create_datapoints(seq, strand, tx_start, tx_end, jn_start):
    """Main feature/label creation routine."""
    seq = 'N'*(CL_max//2) + seq[CL_max//2:-CL_max//2] + 'N'*(CL_max//2)
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')

    tx_start = int(tx_start)
    tx_end = int(tx_end)

    # jn_start is semicolon-separated exon string list
    jn_start = [re.split(';', jn_start)[:-1]]

    A0 = [-np.ones(tx_end - tx_start + 1) for _ in range(1)]
    A1 = [-np.ones(tx_end - tx_start + 1) for _ in range(1)]

    if strand == '+':
        X0 = np.asarray(list(map(int, list(seq))))

        for t in range(1):
            if len(jn_start[t]) > 0:
                A0[t] = np.zeros(tx_end - tx_start + 1)
                A1[t] = np.zeros(tx_end - tx_start + 1)
                for c in jn_start[t]:
                    parts = c.split('|')
                    if len(parts) < 4:
                        continue
                    site_type, cov = parts[3].split(':', 1)
                    coord = parts[1]
                    cov = cov.split(',')
                    assert len(cov) == 1

                    if tx_start <= int(coord) <= tx_end:
                        idx = int(coord) - tx_start
                        A0[t][idx] = get_bin(cov[0], strand, site_type)
                        A1[t][idx] = get_usage(cov[0])

    elif strand == '-':
        X0 = (5 - np.asarray(list(map(int, list(seq[::-1]))))) % 5  # reverse complement

        for t in range(1):
            if len(jn_start[t]) > 0:
                A0[t] = np.zeros(tx_end - tx_start + 1)
                A1[t] = np.zeros(tx_end - tx_start + 1)
                for c in jn_start[t]:
                    parts = c.split('|')
                    if len(parts) < 4:
                        continue
                    site_type, cov = parts[3].split(':', 1)
                    coord = parts[1]
                    cov = cov.split(',')
                    if tx_start <= int(coord) <= tx_end:
                        idx = tx_end - int(coord)
                        A0[t][idx] = get_bin(cov[0], strand, site_type)
                        A1[t][idx] = get_usage(cov[0])

    # If any invalid entries remain
    for t in range(1):
        if np.sum(A0[t] == 5) != 0:
            A0[t] = -np.ones(tx_end - tx_start + 1)

    # Format and encode
    X0_formatted = reformat_data_X(X0, A0)
    A0_original = reformat_data_Y(A0)
    A1_formatted = reformat_data_Y(A1)
    mask = create_mask(A0_original)

    X0_encoded = one_hot_encode_X(X0_formatted)
    A0_encoded = one_hot_encode_Y(A0_original)

    return [X0_encoded, A0_encoded, A1_formatted, mask]


# -------------------------------------------------------------------------
# Misc helper
# -------------------------------------------------------------------------

def clip_datapoints(X, Y, CL, N_GPUS):
    """Ensure datapoints count divisible by N_GPUS and apply context clipping."""
    rem = X.shape[0] % N_GPUS
    clip = (CL_max - CL) // 2

    if rem != 0 and clip != 0:
        return X[:-rem, clip:-clip], [Y[t][:-rem] for t in range(1)]
    elif rem == 0 and clip != 0:
        return X[:, clip:-clip], [Y[t] for t in range(1)]
    elif rem != 0 and clip == 0:
        return X[:-rem], [Y[t][:-rem] for t in range(1)]
    else:
        return X, [Y[t] for t in range(1)]
