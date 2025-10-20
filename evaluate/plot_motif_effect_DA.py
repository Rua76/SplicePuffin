import argparse
import numpy as np
from copy import deepcopy
import torch
import os
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
from numba import njit
from models import SimpleNetModified_DA
import seaborn as sns
import logomaker
import pandas as pd
import networkx as nx
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
@njit
def cross_corr(x, y):
    cors = []
    i = 0
    for j in range(y.shape[1] - 5):
        cors.append(
            np.fmax(
                np.corrcoef(x.flatten(),
                            np.concatenate((y[:, j:], y[:, :j]), axis=1).flatten())[0, 1],
                np.corrcoef(x.flatten(),
                            np.concatenate((y[:, j:], y[:, :j]), axis=1)[::-1, ::-1].flatten())[0, 1],
            )
        )
    return np.array(cors)


def comparemats(mats, mats_norm):
    crossmats = {}
    validmats = {}
    for ii in range(len(mats)):
        for jj in range(ii + 1, len(mats)):
            cors = []
            for i in range(40):
                cors_row = []
                for j in range(40):
                    cors_row.append(cross_corr(mats_norm[ii][i], mats_norm[jj][j]).max())
                cors.append(cors_row)
            cors = np.array(cors)
            crossmats[(ii, jj)] = cors
            validmats[(ii, jj)] = (
                (np.abs(mats_norm[ii]).max(axis=2).max(axis=1)[:, None] > 0.1)
                & (np.abs(mats_norm[jj]).max(axis=2).max(axis=1)[None, :] > 0.1)
            )
    return crossmats, validmats


def plotfun(motifpwm, title=None, ax=None):
    motifpwm = pd.DataFrame(motifpwm, columns=["A", "C", "G", "T"])
    crp_logo = logomaker.Logo(
        motifpwm,
        shade_below=0.5,
        fade_below=0.5,
        font_name="Arial Rounded MT Bold",
        ax=ax,
    )
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=["left", "bottom"], visible=True)
    crp_logo.style_xticks(rotation=90, fmt="%d", anchor=0)
    if title is not None:
        crp_logo.ax.set_title(title)
    crp_logo.ax.set_ylabel("", labelpad=-1)
    crp_logo.ax.xaxis.set_ticks_position("none")
    crp_logo.ax.xaxis.set_tick_params(pad=-1)
    return crp_logo


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze motif similarity and visualize motif effects."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../train/train_parallel",
        help="Directory containing model replicas (default: ./train_parallel)",
    )
    parser.add_argument(
        "--n_models",
        type=int,
        default=12,
        help="Number of model replicas to load (default: 12)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./figures/motif_effects",
        help="Directory to save output figures (default: ./models/figures/motif_effects)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.95,
        help="Threshold for motif similarity (default: 0.95)",
    )
    parser.add_argument(
        "--replicate_select",
        type=int,
        default=1,
        help="Select motifs from this replicate when saving final motif PDF (default: 1)",
    )
    parser.add_argument(
        "--replicate_min",
        type=int,
        default=7,
        help="Minimum number of replicates a motif must appear in to be selected (default: 7)",
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # Load models
    # -------------------------------------------------
    nets = []
    filenames = []
    for i in range(args.n_models):
        f = os.path.join(args.model_dir, f"model.rep{i}.pth")
        net = SimpleNetModified_DA()
        net.load_state_dict(torch.load(f, map_location="cpu"))
        net.cpu()
        nets.append(net)
        filenames.append(f)

    mats = [nets[i].conv.weight.detach().numpy() for i in range(len(nets))]
    mats_norm = [mat - mat.mean(axis=1, keepdims=True) for mat in mats]
    demats_donor = [nets[i].deconv_donor.weight.detach().numpy() for i in range(len(nets))]
    demats_acceptor = [nets[i].deconv_acceptor.weight.detach().numpy() for i in range(len(nets))]

    # Flip spatial dimension
    demats_donor = [mat[..., ::-1] for mat in demats_donor]
    demats_acceptor = [mat[..., ::-1] for mat in demats_acceptor]

    # -------------------------------------------------
    # Compare matrices
    # -------------------------------------------------
    crossmats, validmats = comparemats(mats, mats_norm)

    matchlist = []
    matchscores = []
    for i in range(len(nets)):
        for j in range(i + 1, len(nets)):
            mat = crossmats[(i, j)].copy()
            mat[mat < mat.max(axis=1, keepdims=True)] = 0
            mat[mat < mat.max(axis=0, keepdims=True)] = 0
            mat[~validmats[(i, j)]] = 0
            matchlist.append(
                np.argwhere(mat > args.similarity_threshold).astype(str)
                + np.array(["_" + str(i), "_" + str(j)], dtype=object)[None, :]
            )
            matchscores.append(mat[mat > args.similarity_threshold])

    # -------------------------------------------------
    # Build graph
    # -------------------------------------------------
    sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})
    sns.set_style("white")

    g = nx.Graph()
    g.add_weighted_edges_from(
        np.hstack([np.concatenate(matchlist, axis=0), np.concatenate(matchscores)[:, None]])
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------
    # Generate motif plots
    # -------------------------------------------------
    for cc_i, cc in enumerate(nx.connected_components(g)):
        ncols = 7
        nrows = len(cc)
        fig, axes = plt.subplots(figsize=(20, nrows * 1), nrows=nrows, ncols=ncols, dpi=300)
        if nrows == 1:
            axes = axes[np.newaxis, :]

        for row_i, node in enumerate(cc):
            motifind, matind = map(int, node.split("_"))
            score = np.sum(
                [
                    crossmats[i, matind][:, motifind].max() if i < matind else crossmats[matind, i][motifind, :].max()
                    for i in np.setdiff1d(range(4), [matind])
                ]
            )

            donor = demats_donor[matind][0, motifind]
            acceptor = demats_acceptor[matind][0, motifind]
            x_full = np.arange(-300, 301)

            plotfun(mats[matind][motifind].T, ax=axes[row_i][0])
            axes[row_i][0].set_xticks([])

            # Plot donor/acceptor across ranges
            for idx, bp in enumerate([300, 100, 30]):
                mask = (x_full >= -bp) & (x_full <= bp)
                axes[row_i][1 + 2 * (idx >= 1)].plot(x_full[mask], donor[mask], color="C0")
                axes[row_i][2 + 2 * (idx >= 1)].plot(x_full[mask], acceptor[mask], color="C1")

            sns.despine()

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf_path = os.path.join(args.output_dir, f"motif_cc{cc_i+1}.pdf")
        plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved: {pdf_path}")

    # -------------------------------------------------
    # Select motifs from chosen replicate
    # -------------------------------------------------
    selectmats, selectdemats_donor, selectdemats_acceptor = [], [], []
    for cc in list(nx.connected_components(g)):
        if len(cc) >= args.replicate_min:
            for i in cc:
                motifind, matind = map(int, i.split("_"))
                if matind == args.replicate_select:
                    selectmats.append(mats[matind][motifind])
                    selectdemats_donor.append(demats_donor[matind][:, [motifind], :])
                    selectdemats_acceptor.append(demats_acceptor[matind][:, [motifind], :])

    print("length of selected motifs:", len(selectmats))

    pdf_path = os.path.join(args.output_dir, "motifs_selected.pdf")
    with PdfPages(pdf_path) as pdf:
        for mat in selectmats:
            fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
            plotfun(mat.T, ax=ax)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved all motifs in: {pdf_path}")
