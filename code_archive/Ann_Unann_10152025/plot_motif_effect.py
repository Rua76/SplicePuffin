import numpy as np
from copy import deepcopy

import torch
import os

from torch import nn
from torch.autograd import grad
from torch_fftconv import fft_conv1d, FFTConv1d


class SimpleNetModified(nn.Module):
    def __init__(self, input_channels=4, output_channels=4):
        super(SimpleNetModified, self).__init__()

        self.conv = nn.Conv1d(input_channels, 40, kernel_size=51, padding=25)
        self.activation = nn.Sigmoid()

        # Separate deconv layers for labels (3 channels) and SSE (1 channel)
        self.deconv_labels = FFTConv1d(40, output_channels-1, kernel_size=601, padding=300)  # 3 channels
        self.deconv_SSE = FFTConv1d(40, 1, kernel_size=601, padding=300)  # 1 channel

    def forward(self, x):
        y = self.conv(x)  # Shape: (batch_size, 80, 5000)
        yact = self.activation(y) * y  # Shape: (batch_size, 80, 5000)
        
        # Separate predictions for labels and SSE
        y_pred_label = F.softmax(self.deconv_labels(yact), dim=1)  # Shape: (batch_size, 3, 5000)
        y_pred_SSE = torch.sigmoid(self.deconv_SSE(yact))  # Shape: (batch_size, 1, 5000)
        
        # Concatenate the outputs along channel dimension
        y_pred = torch.cat([y_pred_label, y_pred_SSE], dim=1)  # Shape: (batch_size, 4, 5000)
        
        # Crop the output to match label shape: 5000 -> 4000 by removing 500 from each side
        return y_pred[:, :, 500:-500]  # Final shape: (batch_size, 4, 4000)
    
import glob
nets = []
filenames = []
for i in range(12):
    f = f'./models_4channelOut_1010/models/model.40000.rep'+str(i)+'.pth'
    net = SimpleNetModified()
    net.load_state_dict(torch.load(f, map_location='cpu'))  # Added map_location here
    net.cpu()
    nets.append(net)
    filenames.append(f)
          
mats = [nets[i].conv.weight.detach().numpy() for i in range(len(nets))]
mats_norm = [mat - mat.mean(axis=1, keepdims=True) for mat in mats]
demats_label = [nets[i].deconv_labels.weight.detach().numpy() for i in range(len(nets))]
demats_SSE = [nets[i].deconv_SSE.weight.detach().numpy() for i in range(len(nets))]

from scipy.stats import pearsonr
from scipy.signal import correlate2d
from matplotlib import pyplot as plt

from numba import njit
import random

@njit( )
def cross_corr(x, y):
    cors = []
    i=0
    for j in range(y.shape[1]-5):
        minlen = np.fmin(x.shape[1]-i, y.shape[1]-j)
        cors.append(np.fmax(np.corrcoef(x.flatten(), 
                                 np.concatenate((y[:,j:],y[:,:j]), axis=1).flatten())[0,1],
                           np.corrcoef(x.flatten(), 
                                 np.concatenate((y[:,j:],y[:,:j]), axis=1)[::-1,::-1].flatten())[0,1]))
    return np.array(cors)

def comparemats(mats):
    crossmats = {}
    validmats = {}
    for ii in range(len(mats)):
        for jj in range(ii+1, len(mats)):
            cors = []
            for i in range(40):
                cors_row = []
                for j in range(40):
                    cors_row.append(cross_corr(mats_norm[ii][i], mats_norm[jj][j]).max())
                cors.append(cors_row)
            cors = np.array(cors)
            crossmats[(ii,jj)]=cors
            validmats[(ii,jj)]= (np.abs(mats_norm[ii]).max(axis=2).max(axis=1)[:,None]>0.1) & (np.abs(mats_norm[jj]).max(axis=2).max(axis=1)[None,:]>0.1)

    return crossmats, validmats

crossmats, validmats = comparemats(mats)

matchlist = []
matchscores = []
for i in range(len(nets)):
    for j in range(i+1, len(nets)):
        mat = crossmats[(i, j)].copy()
        mat[mat < mat.max(axis=1, keepdims=True)] = 0
        mat[mat < mat.max(axis=0, keepdims=True)] = 0
        mat[~validmats[(i, j)]] = 0
        # Changed np.object to object here
        matchlist.append(np.argwhere(mat > 0.95).astype(str) + np.array(["_"+str(i), "_"+str(j)], dtype=object)[None, :])
        matchscores.append(mat[mat > 0.95])

import seaborn as sns
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_style("white")

import logomaker
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
prop_cycle = plt.rcParams['axes.prop_cycle']
itercolor =  prop_cycle()
def plotfun(motifpwm, title=None, ax=None):
    motifpwm = pd.DataFrame(motifpwm,columns=['A','C','G','T'])
    crp_logo = logomaker.Logo(motifpwm,
                              shade_below=.5,
                              fade_below=.5,
                              font_name='Arial Rounded MT Bold',
                             ax=ax)

    # style using Logo methods
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)
    if title is not None:
        crp_logo.ax.set_title(title)
    # style using Axes methods
    crp_logo.ax.set_ylabel("", labelpad=-1)
    crp_logo.ax.xaxis.set_ticks_position('none')
    crp_logo.ax.xaxis.set_tick_params(pad=-1)
    return crp_logo

import networkx as nx
from collections import defaultdict
g = nx.Graph()
g.add_weighted_edges_from(np.hstack([np.concatenate(matchlist, axis=0),np.concatenate(matchscores)[:,None]]))

bestmats = []
bestscores = []
filescores = defaultdict(list)

output_dir = "../figures/motif_effects_difference"
os.makedirs(output_dir, exist_ok=True)

for cc_i, cc in enumerate(nx.connected_components(g)):

    # 3 columns: Motif | Label | SSE
    ncols = 3
    nrows = len(cc)
    fig, axes = plt.subplots(figsize=(15, nrows * 1), nrows=nrows, ncols=ncols, dpi=300)

    # Ensure axes is always 2D (handles single-row case)
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row_i, node in enumerate(cc):
        motifind, matind = map(int, node.split('_'))

        # Compute score (for best motif selection)
        score = np.sum([
            crossmats[i, matind][:, motifind].max() if i < matind
            else crossmats[matind, i][motifind, :].max()
            for i in np.setdiff1d(range(4), [matind])
        ])

        # Track best motif matrix for this CC
        if row_i == 0 or score > bestscore:
            bestscore = score
            bestmat = mats[matind][motifind]

        # --- Column 1: Motif heatmap ---
        plotfun(mats[matind][motifind].T, ax=axes[row_i][0])
        axes[row_i][0].set_xticks([])

        # --- Column 2: demats_label (splicing vs non-splicing) ---
        x = np.arange(-300, 301)
        non_splice = demats_label[matind][0, motifind]
        donor = demats_label[matind][1, motifind]
        acceptor = demats_label[matind][2, motifind]

        # Compute differences
        donor_diff = donor - non_splice
        acceptor_diff = acceptor - non_splice

        # Plot both difference curves
        axes[row_i][1].plot(x, donor_diff, label="Donor", color="C0")
        axes[row_i][1].plot(x, acceptor_diff, label="Acceptor", color="C1")

        # --- Column 3: demats_SSE ---
        axes[row_i][2].plot(np.arange(-300, 301), demats_SSE[matind][0, motifind])

        sns.despine()

    # Shared legend and titles
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    axes[0][0].set_title("Motif", fontsize=10)
    axes[0][1].set_title("ΔLabel (vs Non-splice)", fontsize=10)
    axes[0][2].set_title("SSE", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for legend

    # --- Save figure ---
    pdf_path = os.path.join(output_dir, f"motif_cc{cc_i+1}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved: {pdf_path}")

