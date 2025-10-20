import numpy as np
from copy import deepcopy

import torch
import os


from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'

from numba import njit
from models import SimpleNetModified_DA

nets = []
filenames = []
for i in range(12):
    f = f'./train_parallel/model.rep'+str(i)+'.pth'
    net = SimpleNetModified_DA()
    net.load_state_dict(torch.load(f, map_location='cpu'))  # Added map_location here
    net.cpu()
    nets.append(net)
    filenames.append(f)
          
mats = [nets[i].conv.weight.detach().numpy() for i in range(len(nets))]
mats_norm = [mat - mat.mean(axis=1, keepdims=True) for mat in mats]
demats_donor = [nets[i].deconv_donor.weight.detach().numpy() for i in range(len(nets))]
demats_acceptor = [nets[i].deconv_acceptor.weight.detach().numpy() for i in range(len(nets))]

# Flip the spatial dimension (kernel axis)
demats_donor = [mat[..., ::-1] for mat in demats_donor]
demats_acceptor   = [mat[..., ::-1] for mat in demats_acceptor]


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

output_dir = "./models/figures/motif_effects"
os.makedirs(output_dir, exist_ok=True)

for cc_i, cc in enumerate(nx.connected_components(g)):
    # 7 columns: Motif | Label * 3 | SSE
    ncols = 7
    nrows = len(cc)
    fig, axes = plt.subplots(figsize=(20, nrows * 1), nrows=nrows, ncols=ncols, dpi=300)

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
        
        donor = demats_donor[matind][0, motifind]
        acceptor = demats_acceptor[matind][0, motifind]
        x_full = np.arange(-300, 301)
        
        # Column 2: full range
        axes[row_i][1].plot(x_full, donor, label="Donor", color="C0")
        axes[row_i][2].plot(x_full, acceptor, label="Acceptor", color="C1")

        # Column 3: ±100 bp
        mask_100 = (x_full >= -100) & (x_full <= 100)
        axes[row_i][3].plot(x_full[mask_100], donor[mask_100], color="C0")
        axes[row_i][4].plot(x_full[mask_100], acceptor[mask_100], color="C1")

        # Column 4: ±30 bp
        mask_30 = (x_full >= -30) & (x_full <= 30)
        axes[row_i][5].plot(x_full[mask_30], donor[mask_30], color="C0")
        axes[row_i][6].plot(x_full[mask_30], acceptor[mask_30], color="C1")

        sns.despine()

    axes[0][0].set_title("Motif", fontsize=10)
    axes[0][1].set_title("Donor layer", fontsize=10)
    axes[0][2].set_title("Acceptor layer", fontsize=10)
    axes[0][3].set_title("Donor layer +-100bp center", fontsize=10)
    axes[0][4].set_title("Acceptor layer +-100bp center", fontsize=10)
    axes[0][5].set_title("Donor layer +-30bp center", fontsize=10)
    axes[0][6].set_title("Acceptor layer +-30bp center", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for legend

    # --- Save figure ---
    pdf_path = os.path.join(output_dir, f"motif_cc{cc_i+1}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved: {pdf_path}")

selectmats = []
selectdemats_donor = []
selectdemats_acceptor = []

for cc in list(nx.connected_components(g)):
    if len(cc)>=1:  # select only motifs appearing in at least 2 replicates
        for i in cc:
            motifind, matind = list(map(int, i.split('_')))

            if matind == 1: # select motifs from replicate 7 as an example
                selectmats.append(mats[matind][motifind])
                selectdemats_donor.append(demats_donor[matind][:,[motifind],:])
                selectdemats_acceptor.append(demats_acceptor[matind][:,[motifind],:])
 
print('length of selected motifs: ',len(selectmats))

from matplotlib.backends.backend_pdf import PdfPages

pdf_path = os.path.join(output_dir, "motifs_1replicates.pdf")
with PdfPages(pdf_path) as pdf:
    for ii, mat in enumerate(selectmats):
        fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=1, dpi=300)
        plotfun(mat.T, ax=axes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"Saved all motifs in: {pdf_path}")
