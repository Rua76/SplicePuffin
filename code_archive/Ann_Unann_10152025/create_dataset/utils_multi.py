###############################################################################
'''This code has functions which process the information in the .h5 files
datafile_{}_{}.h5 and convert them into a format usable by Keras.'''
###############################################################################

import numpy as np
import re
from math import ceil
from constants import *

assert CL_max % 2 == 0

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
# One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
# to A, C, G, T respectively.

OUT_MAP = np.asarray([[1, 0, 0], # no splicing
                      [0, 1, 0], # donor
                      [0, 0, 1], #  acceptor
                      [0, 0, 0], # missing
                ])

def reformat_data_X(X0, Y0):
    # This function converts X0, Y0 of the create_datapoints function into
    # blocks such that the data is broken down into data points where the
    # input is a sequence of length SL+CL_max corresponding to SL nucleotides
    # of interest and CL_max context nucleotides, the output is a sequence of
    # length SL corresponding to the splicing information of the nucleotides
    # of interest. The CL_max context nucleotides are such that they are
    # CL_max/2 on either side of the SL nucleotides of interest.

    assert len(X0) == len(Y0[0]) + CL_max
    num_points = ceil_div(len(Y0[0]), SL)

    Xd = np.zeros((num_points, SL+CL_max))
    X0 = np.pad(X0, [0, SL], 'constant', constant_values=0)

    for i in range(num_points):
        Xd[i] = X0[SL*i:CL_max+SL*(i+1)]

    return Xd

def reformat_data_Y(Y0):
    num_points = ceil_div(len(Y0[0]), SL)
    Yd = [-np.ones((num_points, SL)) for t in range(1)]

    Y0 = [np.pad(Y0[t], [0, SL], 'constant', constant_values=-1) for t in range(1)]
 
    for t in range(1):
        for i in range(num_points):
            Yd[t][i] = Y0[t][SL*i:SL*(i+1)]

    return Yd

def one_hot_encode_X(Xd):
    return IN_MAP[Xd.astype('int8')]

def one_hot_encode_Y(Yd):
    return [OUT_MAP[Yd[t].astype('int8')] for t in range(1)]

def ceil_div(x, y):
    return int(ceil(float(x)/y))

# -3 = unexpressed, -2 = # of reads >0 but <threshold, -1 = spliced but no usage est
def get_usage(cov):
    if cov == "":
        return 0
    cov = float(cov)
    if cov < -2.99: # -3
        return -1
    elif cov < -1.99 or cov == 0.0: # -2 or 0
        return 0
    elif cov > -1.01 and cov < -0.99: # -1
        return -1
    else:
        return cov

def get_bin(cov, strand=None, site_type=None):
    if cov == "":
        return 0
    cov = float(cov)
    if cov < -2.99:
        return 5  # no experssion (index 5 in OUT_MAP)
    elif cov < -1.99:
        return 0  # no splicing (index 0 in OUT_MAP)
    else:
        # Only classify if strand and site_type are provided
        if strand is not None and site_type is not None:
            if site_type == 'A':  # Acceptor
                return 2  # acceptor (index 2 in OUT_MAP)
            elif site_type == 'D':  # Donor
                return 1  # donor (index 1 in OUT_MAP)
        return 0  # default to no splicing if not classified

def create_datapoints(seq, strand, tx_start, tx_end, jn_start):
    seq = 'N'*(CL_max//2) + seq[CL_max//2:-CL_max//2] + 'N'*(CL_max//2)
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')

    tx_start = int(tx_start)
    tx_end = int(tx_end) 

    jn_start = map(lambda x: re.split(';', x)[:-1], [jn_start])
    

    A0 = [-np.ones(tx_end-tx_start+1) for t in range(1)]
    A1 = [-np.ones(tx_end-tx_start+1) for t in range(1)]

    if strand == '+':
        X0 = np.asarray(map(int, list(seq)))
 
        for t in range(1):
            if len(jn_start[t]) > 0:
                A0[t] = np.zeros(tx_end-tx_start+1)
                A1[t] = np.zeros(tx_end-tx_start+1)
                for c in jn_start[t]:
                    parts = c.split('|')
                    site_type, cov = parts[3].split(':') # Skip the type and ':'
                    
                    coord = parts[1]
                    cov = cov.split(',')
                    assert(len(cov)==1)
                    
                    if tx_start <= int(coord) <= tx_end:
                        idx = int(coord)-tx_start
                        A0[t][idx] = get_bin(cov[0], strand, site_type)
                        A1[t][idx] = get_usage(cov[0])

                     
    elif strand == '-':
        X0 = (5-np.asarray(map(int, list(seq[::-1])))) % 5  # Reverse complement

        for t in range(1):
            if len(jn_start[t]) > 0:
                A0[t] = np.zeros(tx_end-tx_start+1)
                A1[t] = np.zeros(tx_end-tx_start+1)

                for c in jn_start[t]:
                    parts = c.split('|')
                    site_type, cov = parts[3].split(':') # Skip the type and ':'
                    coord = parts[1]
                    cov = cov.split(',')
                    if tx_start <= int(coord) <= tx_end:
                        idx = tx_end-int(coord)
                        A0[t][idx] = get_bin(cov[0], strand, site_type)
                        A1[t][idx] = get_usage(cov[0])

    if np.sum(A0[t]==5) != 0:
        A0[t] = -np.ones(tx_end-tx_start+1)


    X0 = reformat_data_X(X0, A0)
    A0 = reformat_data_Y(A0)
    A1 = reformat_data_Y(A1)


    X0 = one_hot_encode_X(X0)
    A0 = one_hot_encode_Y(A0)


    return [X0, A0, A1]

def clip_datapoints(X, Y, CL, N_GPUS):
    # This function is necessary to make sure of the following:
    # (i) Each time model_m.fit is called, the number of datapoints is a
    # multiple of N_GPUS. Failure to ensure this often results in crashes.
    # (ii) If the required context length is less than CL_max, then
    # appropriate clipping is done below.
    # Additionally, Y is also converted to a list (the .h5 files store 
    # them as an array).

    rem = X.shape[0]%N_GPUS
    clip = (CL_max-CL)//2

    if rem != 0 and clip != 0:
        return X[:-rem, clip:-clip], [Y[t][:-rem] for t in range(1)]
    elif rem == 0 and clip != 0:
        return X[:, clip:-clip], [Y[t] for t in range(1)]
    elif rem != 0 and clip == 0:
        return X[:-rem], [Y[t][:-rem] for t in range(1)]
    else:
        return X, [Y[t] for t in range(1)]