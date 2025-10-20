#!/usr/bin/env python3
"""
plot_predictions_vs_targets_NDA.py
----------------------------------
Plot model predictions vs true targets for visualization.

Usage:
    python plot_predictions_vs_targets_NDA.py \
        --model_path ./trained_models/model.rep0.pth \
        --test_data ../../create_dataset/dataset_test_1.h5 \
        --sample_idx 0 \
        --output_dir ./plots
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_fftconv import FFTConv1d
from torch import nn
import torch.nn.functional as F
import h5py
from matplotlib.patches import Rectangle  # Add this import
import gffutils
from models import SimpleNetModified_DA

def load_sample_from_h5(file_path, sample_idx):
    """Load a single sample from H5 file"""
    with h5py.File(file_path, 'r') as data:
        X = torch.Tensor(data['X' + str(sample_idx)][:].T)  # Shape: (4, 4000)
        Y = torch.Tensor(data['Y' + str(sample_idx)][:].T)  # Shape: (4, 4000)
        Z = data['Z' + str(sample_idx)][:].T
        decode_Z =[z.decode('utf-8') for z in Z]
    return X, Y, decode_Z

def find_peaks(signal, threshold=0.5, min_distance=10):
    """Find peaks in 1D signal above threshold with minimum distance"""
    peaks = []
    for i in range(len(signal)):
        if signal[i] > threshold:
            # Check if this is a local maximum within the neighborhood
            left = max(0, i - min_distance)
            right = min(len(signal), i + min_distance + 1)
            if signal[i] == max(signal[left:right]):
                peaks.append(i)
    return peaks

def get_genes(chr, start, end, gtf):
    # Query a small region around pos (±GENE_WINDOW) to capture nearby genes
    genes = gtf.region((chr, start, end), featuretype="gene")
    genes_pos, genes_neg = {}, {}

    for gene in genes:
        gene_start, gene_end = gene[3], gene[4]

        # keep gene if pos falls within windowed range
        if not (gene_start-5000 <= start <= end <= gene_end +5000):
            continue

        gene_id = gene["gene_id"][0]
        exons = []
        for exon in gtf.children(gene, featuretype="exon"):
            exons.extend([exon[3], exon[4]])

        if gene[6] == '+':
            genes_pos[gene_id] = exons
        elif gene[6] == '-':
            genes_neg[gene_id] = exons

    return (genes_pos, genes_neg)


def plot_predictions_vs_targets(predictions, targets, sample_idx, output_path, gtf, threshold=0, coord=None):
    """
    Plot model predictions vs true targets for both channels and include gene/exon annotation.

    Args:
        predictions: numpy array of shape (4, L) - model predictions
        targets: numpy array of shape (4, L) - ground truth
        sample_idx: index of the sample for title
        output_path: path to save the plot
        gtf: loaded GTF file (pybedtools or gffutils object)
        threshold: threshold for peak detection
        coord: tuple or list like ('chr1', 100000, 104000, '+')
    """

    # === Parse coordinates ===
    chromosome = str(coord[0])
    strand = coord[-1]
    if strand == '+':
        start = int(coord[1])+500
        end = int(coord[2])-500
    else:
        start = int(coord[2])+500
        end = int(coord[1])-500

    # flip if on negative strand
    if strand == '-':
        predictions = np.flip(predictions, axis=1)
        targets = np.flip(targets, axis=1)
    
    # === Get genes and exons ===
    genes_pos, genes_neg = get_genes(chromosome, start, end, gtf)
    genes_dict = genes_pos if strand == '+' else genes_neg

    # === Create figure ===
    fig, axes = plt.subplots(3, 1, figsize=(20, 9), sharex=True)
    channel_names = ['Gene body', 'Donor', 'Acceptor']
    
    # === Plot gene track (top panel) ===
    ax_gene = axes[0]
    ax_gene.set_ylabel('Gene', fontsize=10)
    ax_gene.set_yticks([])
    ax_gene.set_xlim(start, end)
    ax_gene.set_title(f"Gene structure ({chromosome}:{start}-{end} {strand})", fontsize=12, fontweight='bold')
    
    exon_count = 0
    for gene_id, gene_exons in genes_dict.items():
        for i in range(0, len(gene_exons), 2):
            exon_start = gene_exons[i]
            exon_end = gene_exons[i + 1]
            if exon_end < start or exon_start > end:
                continue
            draw_exon_start = max(start, exon_start)
            draw_exon_end = min(end, exon_end)
            exon_width = draw_exon_end - draw_exon_start
            if exon_width > 0:
                y_pos = -0.2
                exon_height = 1
                exon_rect = Rectangle(
                    (draw_exon_start, y_pos), exon_width, exon_height, 
                    facecolor='blue', alpha=0.7, edgecolor='black'
                )
                ax_gene.add_patch(exon_rect)
                exon_count += 1
    print(f"  Total exons drawn: {exon_count}")

    # Add strand arrow
    strand_symbol = '→' if strand == '+' else '←'
    ax_gene.text(end + (end - start) * 0.01, 0, strand_symbol, fontsize=14, ha='left', va='center')

    # === Plot predictions and targets ===
    for channel in range(2):
        ax = axes[channel+1]
        pred = predictions[channel, :]
        target = targets[channel, :]

        positions = np.linspace(start, end, len(pred))
        
        ax.plot(positions, pred, 'b-', alpha=0.7, linewidth=1, label='Prediction')

        target_peaks = np.where(target > 0)[0]
        if len(target_peaks) > 0:
            ax.scatter(positions[target_peaks], target[target_peaks], 
                       color='red', s=30, marker='v', label='Target peaks', zorder=5)

        ax.set_title(f'{channel_names[channel+1]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability')
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    axes[-1].set_xlabel(f'Genomic Position ({chromosome})')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', format='pdf')
    plt.close()

    print(f"Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot model predictions vs targets with gene annotations")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--test_data", required=True, help="Path to test dataset (.h5)")
    parser.add_argument("--gtf", required=True, help="Path to GTF annotation file")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to plot")
    parser.add_argument("--output_dir", required=True, help="Directory to save plots")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for peak detection")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to plot")
    
    args = parser.parse_args()

    # === Create output directory ===
    os.makedirs(args.output_dir, exist_ok=True)

    # === Load GTF annotation ===
    try:
        gtf = gffutils.FeatureDB(args.gtf)
        print(f"[DEBUG] Loaded annotation database: {args.gtf}")
    except Exception as e:
        print(f"[ERROR] Failed loading annotation file {args.gtf}: {e}")
        exit(1)

    # === Load model ===
    print(f"Loading model from {args.model_path}")
    model = SimpleNetModified_DA(input_channels=4)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.cuda()

    # === Plot multiple samples ===
    for sample_idx in range(args.sample_idx, args.sample_idx + args.num_samples):
        print(f"\n--- Processing sample {sample_idx} ---")
        
        # Load sample
        X, Y, coord = load_sample_from_h5(args.test_data, sample_idx)
        print(f"Sample {sample_idx} loaded: coord={coord}")
        
        # Run model prediction
        with torch.no_grad():
            X_input = X.unsqueeze(0).cuda()
            pred = model(X_input)
            pred_np = pred.squeeze(0).cpu().numpy()
            target_np = Y.numpy()
        
        # Plot results
        output_path = os.path.join(args.output_dir, f"sample_{sample_idx}.pdf")
        plot_predictions_vs_targets(
            pred_np,
            target_np,
            sample_idx,
            output_path,
            gtf=gtf,
            threshold=args.threshold,
            coord=coord
        )
        
        # Print statistics
        print(f"Sample {sample_idx} statistics:")
        for channel in range(2):
            pred_peaks = find_peaks(pred_np[channel], threshold=args.threshold)
            target_peaks = np.where(target_np[channel] > 0.5)[0]
            print(f"  Channel {channel}: {len(pred_peaks)} pred peaks, {len(target_peaks)} target peaks")

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()