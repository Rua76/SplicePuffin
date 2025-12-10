#!/usr/bin/env python3
"""
evaluate_connor_testset_scatter.py

Evaluate splice site prediction model on exon dataset with scatter plots of AUPRC vs. width bins.
"""

import os
import random
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns

# Import models - handle both 2-layer and 3-layer simple models
try:
    from models import SimpleNetModified_DA_SSE, SimpleNetModified_DA, SimpleNetModified_DA_TripleLayers
    SIMPLE_MODELS_AVAILABLE = True
except ImportError:
    SIMPLE_MODELS_AVAILABLE = False
    print("Warning: Simple models module not available. Only SpliceAI will be available.")

from utils import one_hot_encode_sequence
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from tensorflow.keras.models import load_model
import pyfastx
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# -------------------------
# Configuration
# -------------------------

warnings.simplefilter(action='ignore', category=FutureWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REMOVE_PADDING = 240  # input padding to remove

# -------------------------
# Helper Functions
# -------------------------

def parse_code(code):
    """Parse the complex code field into structured attributes."""
    parts = code.split('_')
    return {
        'FIL': parts[0].split(':')[0],
        'GC_bin': parts[0].split(':')[1],
        'Wi_bin': parts[1].split(':')[1],
        'Coding': parts[2],
        'Strand': parts[3],
        'RH': parts[4],
        'Uniq': parts[5]
    }

def load_model_by_tag(model_tag, model_path, model_type=None):
    """
    Load the appropriate model based on tag and path.
    
    Args:
        model_tag: Unique identifier for the model
        model_path: Path to model checkpoint
        model_type: Type of model ('simple_2layer', 'simple_3layer', 'simple_sse', 'spliceai')
    
    Returns:
        (model_object, model_type, save_path)
    """
    if not os.path.exists(model_path):
        print(f"[Warning] Model checkpoint not found for {model_tag}: {model_path}")
        return None, None, None
    
    print(f"[Loading model] {model_tag}: {model_path}")
    
    if model_type is None:
        # Auto-detect model type from tag
        if 'spliceai' in model_tag.lower():
            model_type = 'spliceai'
        elif 'sse' in model_tag.lower():
            model_type = 'simple_sse'
        elif 'triple' in model_tag.lower() or '3layer' in model_tag.lower():
            model_type = 'simple_3layer'
        else:
            model_type = 'simple_2layer'
    
    if model_type == 'spliceai':
        model = load_model(model_path, compile=False)
        return model, model_type, model_path
    elif model_type == 'simple_binary':
        if not SIMPLE_MODELS_AVAILABLE:
            raise ImportError("Simple models module not available")
        model = SimpleNetModified_DA(input_channels=4)
    elif model_type == 'simple_3layer':
        if not SIMPLE_MODELS_AVAILABLE:
            raise ImportError("Simple models module not available")
        model = SimpleNetModified_DA_TripleLayers(input_channels=4)
    elif model_type == 'simple_2layer':
        if not SIMPLE_MODELS_AVAILABLE:
            raise ImportError("Simple models module not available")
        model = SimpleNetModified_DA_SSE(input_channels=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, model_type, model_path

def generate_prediction_summary(df, npz_dir, save_path, threshold=0.05):
    """
    Merge predictions from saved .npz files back into original dataframe.
    Compute TP/TN/FP/FN metrics at BASE RESOLUTION.
    Filter out bait FP labels for Fi/La exons during loading but preserve bait positions for score analysis.
    Each exon has 0 or 1 true splice site per type.
    """
    
    model_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
    merged = df.copy()
    columns_to_drop = ['seq', 'three_SS_seq', 'five_SS_seq', 'three_SS_position', 'five_SS_position', 'fiveSS_score'] # remove dupliacte 5ss_score
    merged = merged.drop(columns=[col for col in columns_to_drop if col in merged.columns])

    # Ensure the key columns in the original dataframe have the right data types
    merged['start'] = merged['start'].astype(int)
    merged['end'] = merged['end'].astype(int)
    merged['seqnames'] = merged['seqnames'].astype(str)
    merged['strand'] = merged['strand'].astype(str)
    merged['code'] = merged['code'].astype(str)

    for model_file in model_files:
        model_tag = model_file.replace("_pred_label_pairs.npz", "")
        data = np.load(os.path.join(npz_dir, model_file), allow_pickle=True)
        preds, labels, meta_list = data['preds'], data['labels'], data['meta']
        
        summary_rows = []
        
        for i, meta in enumerate(meta_list):
            chrom = str(meta['chrom'])
            start = int(meta['start'])  # This should now be numeric
            end = int(meta['end'])    # This should now be numeric  
            strand = str(meta['strand'])
            code = str(meta['code'])
            
            # Get predictions and labels for this sequence - handle list of arrays
            seq_pred = preds[i]  # This should be a 2D array [seq_len, 2]
            seq_label = labels[i]  # This should be a 2D array [seq_len, 2]
            
            # Apply threshold to get binary predictions for this sequence
            seq_pred_binary = (seq_pred >= threshold).astype(int)
            
            seq_pred_5ss = seq_pred_binary[:, 0]  # 5SS binary predictions
            seq_pred_3ss = seq_pred_binary[:, 1]  # 3SS binary predictions
            seq_raw_pred_5ss = seq_pred[:, 0]     # 5SS raw scores
            seq_raw_pred_3ss = seq_pred[:, 1]     # 3SS raw scores
            seq_label_5ss = seq_label[:, 0].copy()  # Copy to avoid modifying original
            seq_label_3ss = seq_label[:, 1].copy()  # Copy to avoid modifying original
            
            # Find ORIGINAL true positions (0 or 1 per exon)
            original_true_5ss_positions = np.where(seq_label[:, 0] == 1)[0]
            original_true_3ss_positions = np.where(seq_label[:, 1] == 1)[0]
            
            # Get single score at true position (or NaN if no true site)
            true_5ss_score = seq_raw_pred_5ss[original_true_5ss_positions[0]] if len(original_true_5ss_positions) > 0 else np.nan
            true_3ss_score = seq_raw_pred_3ss[original_true_3ss_positions[0]] if len(original_true_3ss_positions) > 0 else np.nan
            
            # FILTER BAIT FP LABELS: Remove invalid splice site labels for boundary exons
            if code == "Fi":  # First exon - remove all 3SS labels (no upstream 3SS)
                seq_label_3ss[:] = 0
                
            if code == "La":  # Last exon - remove all 5SS labels (no downstream 5SS)
                seq_label_5ss[:] = 0
            
            # Find positions of true splice sites (after filtering)
            true_5ss_positions = np.where(seq_label_5ss == 1)[0]
            true_3ss_positions = np.where(seq_label_3ss == 1)[0]
            
            # Find predicted positions
            pred_5ss_positions = np.where(seq_pred_5ss == 1)[0]
            pred_3ss_positions = np.where(seq_pred_3ss == 1)[0]
            
            # Compute base-resolution metrics for 5SS (using filtered labels)
            tp_5ss = len(set(true_5ss_positions) & set(pred_5ss_positions))
            fp_5ss = len(set(pred_5ss_positions) - set(true_5ss_positions))
            fn_5ss = len(set(true_5ss_positions) - set(pred_5ss_positions))
            tn_5ss = len(seq_pred_5ss) - (tp_5ss + fp_5ss + fn_5ss)
            
            # Compute base-resolution metrics for 3SS (using filtered labels)
            tp_3ss = len(set(true_3ss_positions) & set(pred_3ss_positions))
            fp_3ss = len(set(pred_3ss_positions) - set(true_3ss_positions))
            fn_3ss = len(set(true_3ss_positions) - set(pred_3ss_positions))
            tn_3ss = len(seq_pred_3ss) - (tp_3ss + fp_3ss + fn_3ss)
            
            summary_rows.append({
                "seqnames": str(chrom),  # Ensure string type
                "start": int(start),     # Ensure int type
                "end": int(end),         # Ensure int type
                "strand": str(strand),   # Ensure string type
                "code": str(code),       # Ensure string type
                f"TP_5SS_{model_tag}": tp_5ss,
                f"TN_5SS_{model_tag}": tn_5ss,
                f"FP_5SS_{model_tag}": fp_5ss,
                f"FN_5SS_{model_tag}": fn_5ss,
                f"TP_3SS_{model_tag}": tp_3ss,
                f"TN_3SS_{model_tag}": tn_3ss,
                f"FP_3SS_{model_tag}": fp_3ss,
                f"FN_3SS_{model_tag}": fn_3ss,
                f"pred_5ss_sites_count_{model_tag}": len(pred_5ss_positions),
                f"pred_3ss_sites_count_{model_tag}": len(pred_3ss_positions),
                # Store single scores instead of lists
                f"prediction_at_5ss_{model_tag}": true_5ss_score,
                f"prediction_at_3ss_{model_tag}": true_3ss_score,
            })
        
        pred_df = pd.DataFrame(summary_rows)
        
        # Ensure consistent data types in the prediction dataframe
        pred_df['start'] = pred_df['start'].astype(int)
        pred_df['end'] = pred_df['end'].astype(int)
        pred_df['seqnames'] = pred_df['seqnames'].astype(str)
        pred_df['strand'] = pred_df['strand'].astype(str)
        pred_df['code'] = pred_df['code'].astype(str)
        
        # Use the correct column names that match the original dataframe
        merged = merged.merge(pred_df, on=["seqnames", "start", "end", "strand", "code"], how="left")
    
    merged.to_csv(save_path, sep="\t", index=False)
    print(f"[Saved base-resolution prediction summary with threshold {threshold}] {save_path}")
    
    return merged

# -------------------------
# Dataset Class
# -------------------------

class SpliceDataset(Dataset):
    def __init__(self, df, fasta_obj=None, is_spliceai=False, padding=5010):
        """
        df: dataframe with exon info
        fasta_obj: pyfastx.Fasta object (required if is_spliceai=True)
        is_spliceai: if True, fetch sequences from fasta using genomic coordinates
        padding: number of bp upstream/downstream to include
        """
        self.df = df
        self.fasta = fasta_obj
        self.is_spliceai = is_spliceai
        self.padding = padding

        if self.is_spliceai and self.fasta is None:
            raise ValueError("Fasta object must be provided when is_spliceai=True")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chrom = row['seqnames']
        start = int(row['start']) - 1  # convert to 0-based
        end = int(row['end'])
        strand = row['strand']

        # -------------------------------
        # Fetch or extract the sequence
        # -------------------------------
        if self.is_spliceai:
            seq_start = max(0, start - self.padding)
            seq_end = end + self.padding
            seq = self.fasta[chrom][seq_start:seq_end]
            seq = seq.seq.upper() if strand == '+' else seq.antisense.upper()
        else:
            # Use provided sequence (trim padding)
            seq = row['seq'][REMOVE_PADDING:-REMOVE_PADDING]

        # One-hot encode
        enc = one_hot_encode_sequence(seq)

        # -------------------------------
        # Merge metadata
        # -------------------------------
        # Base metadata
        meta = {
            "chrom": chrom,
            "start": row['start'],
            "end": row['end'],
            "strand": strand,
            "code": row.get("code", ""),
            "width": row.get("width", 0),
            "three_SS_seq": row.get("three_SS_seq", ""),
            "five_SS_seq": row.get("five_SS_seq", ""),
            "three_SS_score": row.get("three_SS_score", None),
            "five_SS_score": row.get("five_SS_score", None),
        }

        # Parsed code metadata
        parsed_meta = parse_code(row['code']) if 'code' in row and not pd.isna(row['code']) else {}
        meta.update(parsed_meta)  # Merge parsed code info (e.g., exon rank, transcript_id, etc.)

        return {
            "seq": torch.tensor(enc),
            "meta": meta,
            "id": idx,
            "raw_seq": seq
        }

# ---------------------------------------------------------
# AUPRC calculation functions
# ---------------------------------------------------------

def compute_auprc_by_width(preds_all, labels_all, meta_all, site_type="5ss", width_bin_size=50):
    """
    Compute AUPRC for each width bin.
    
    Args:
        preds_all: List of prediction arrays
        labels_all: List of label arrays
        meta_all: List of metadata dictionaries
        site_type: "5ss" or "3ss"
        width_bin_size: Size of width bins in base pairs
        
    Returns:
        width_bins: Array of bin centers
        auprc_values: Array of AUPRC values for each bin
        bin_counts: Array of sample counts in each bin
    """
    # Determine channel index
    ch = 0 if site_type == "5ss" else 1
    
    # Collect all widths and corresponding predictions/labels
    widths = []
    y_true_list = []
    y_score_list = []
    
    for preds, labels, meta in zip(preds_all, labels_all, meta_all):
        width = int(meta.get('width', 0))
        if width > 0:
            widths.append(width)
            y_true_list.extend(labels[:, ch].tolist())
            y_score_list.extend(preds[:, ch].tolist())
    
    if not widths:
        return np.array([]), np.array([]), np.array([])
    
    widths = np.array(widths)
    y_true_all = np.array(y_true_list)
    y_score_all = np.array(y_score_list)
    
    # Create width bins
    max_width = int(np.ceil(np.max(widths) / width_bin_size) * width_bin_size)
    n_bins = max_width // width_bin_size
    
    width_bins = []
    auprc_values = []
    bin_counts = []
    
    for bin_idx in range(n_bins):
        bin_start = bin_idx * width_bin_size
        bin_end = (bin_idx + 1) * width_bin_size
        bin_center = (bin_start + bin_end) / 2
        
        # Find samples in this width bin
        bin_mask = (widths >= bin_start) & (widths < bin_end)
        
        if np.sum(bin_mask) > 0:
            # For AUPRC calculation, we need to group by sample
            # Extract predictions and labels for samples in this bin
            bin_y_true = []
            bin_y_score = []
            
            # We need to reconstruct the grouped data
            sample_idx = 0
            for i, (preds, labels, meta) in enumerate(zip(preds_all, labels_all, meta_all)):
                width = int(meta.get('width', 0))
                if (width >= bin_start) and (width < bin_end):
                    bin_y_true.extend(labels[:, ch].tolist())
                    bin_y_score.extend(preds[:, ch].tolist())
                sample_idx += 1
            
            bin_y_true = np.array(bin_y_true)
            bin_y_score = np.array(bin_y_score)
            
            if len(bin_y_true) > 0 and np.sum(bin_y_true) > 0:
                precision, recall, _ = precision_recall_curve(bin_y_true, bin_y_score)
                auprc = auc(recall, precision)
            else:
                auprc = np.nan
            
            width_bins.append(bin_center)
            auprc_values.append(auprc)
            bin_counts.append(np.sum(bin_mask))
    
    return np.array(width_bins), np.array(auprc_values), np.array(bin_counts)

def compute_auprc_maps(preds_all, labels_all, meta_all, FIL_order):
    auprc_maps = {fil: {"5ss": {}, "3ss": {}} for fil in FIL_order}

    for preds, labels, meta in zip(preds_all, labels_all, meta_all):
        Wi_bin, GC_bin, fil = meta['Wi_bin'], meta['GC_bin'], meta['FIL']
        if fil not in FIL_order:
            continue

        for site, ch in zip(["5ss", "3ss"], [0, 1]):
            key = (Wi_bin, GC_bin)
            d = auprc_maps[fil][site].setdefault(key, {'y_true': [], 'y_score': []})
            d['y_true'].extend(labels[:, ch].tolist())
            d['y_score'].extend(preds[:, ch].tolist())

    return auprc_maps

# ---------------------------------------------------------
# Scatter plot functions
# ---------------------------------------------------------
def compute_auprc_by_width(preds_all, labels_all, meta_all, site_type="5ss", n_bins=50):
    ch = 0 if site_type == "5ss" else 1
    
    widths = []
    for meta in meta_all:
        width = int(meta.get('width', 0))
        if width > 0:
            widths.append(width)
    
    if not widths:
        return np.array([]), np.array([]), np.array([]), [], np.array([])

    widths = np.array(widths)

    if len(widths) < n_bins:
        n_bins = len(widths)
        if n_bins == 0:
            return np.array([]), np.array([]), np.array([]), [], np.array([])

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(widths, percentiles))
    if len(bin_edges) < 2:
        return np.array([]), np.array([]), np.array([]), [], np.array([])

    bin_indices = []
    auprc_values = []
    bin_counts = []
    width_ranges = []
    median_widths = []

    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        bin_y_true = []
        bin_y_score = []
        bin_widths = []
        bin_count = 0

        for preds, labels, meta in zip(preds_all, labels_all, meta_all):
            width = int(meta.get('width', 0))
            if bin_start <= width < bin_end:
                bin_widths.append(width)
                bin_y_true.extend(labels[:, ch].tolist())
                bin_y_score.extend(preds[:, ch].tolist())
                bin_count += 1

        if bin_count > 0:
            # AUPRC
            if np.sum(bin_y_true) > 0:
                precision, recall, _ = precision_recall_curve(
                    np.array(bin_y_true), np.array(bin_y_score)
                )
                auprc = auc(recall, precision)
            else:
                auprc = np.nan

            bin_indices.append(i + 1)
            auprc_values.append(auprc)
            bin_counts.append(bin_count)
            width_ranges.append((bin_start, bin_end))
            median_widths.append(np.median(bin_widths))

    return (
        np.array(bin_indices),
        np.array(auprc_values),
        np.array(bin_counts),
        width_ranges,
        np.array(median_widths),
    )

def plot_auprc_scatter_by_fil(models_data, save_dir=None, figsize=(15, 10)):
    """
    Plot scatter plots of AUPRC vs. width bins for multiple models, split by FIL.
    
    Args:
        models_data: List of tuples (model_tag, preds_all, labels_all, meta_all)
        save_dir: Directory to save the plot
        figsize: Figure size
    """
    FIL_order = ["Fi", "In", "La"]
    FIL_labels = {"Fi": "First", "In": "Internal", "La": "Last"}
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    n_bins = 50  # Exactly 50 width bins
    
    # Color palette for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
    
    for row_idx, fil in enumerate(FIL_order):
        for col_idx, site_type in enumerate(["5ss", "3ss"]):
            ax = axes[row_idx, col_idx]
            
            ax.set_xlabel("Width Bin (1-50)")
            if col_idx == 0:
                ax.set_ylabel(f"{FIL_labels[fil]} Exon\nAUPRC")
            ax.set_title(f"{FIL_labels[fil]} Exon - {site_type.upper()}")
            ax.grid(True, alpha=0.3)
            
            # Set x-axis ticks and limits
            ax.set_xticks(np.arange(0, 51, 5))  # Show every 5th bin from 0 to 50
            ax.set_xlim(0, 51)
            
            # Set y-axis limits to 0-1
            #ax.set_ylim(0, 1)
            #ax.set_yticks(np.arange(0, 1.1, 0.1))  # Show ticks every 0.1
            
            for model_index, ((model_tag, preds_all, labels_all, meta_all), color) in enumerate(zip(models_data, colors)):
                # Filter data for this FIL type
                filtered_data = []
                for preds, labels, meta in zip(preds_all, labels_all, meta_all):
                    if meta.get('FIL', '') == fil:
                        filtered_data.append((preds, labels, meta))

                if filtered_data:
                    filt_preds, filt_labels, filt_meta = zip(*filtered_data)
                    bin_indices, auprc_values, bin_counts, width_ranges, median_widths = compute_auprc_by_width(
                        list(filt_preds), list(filt_labels), list(filt_meta),
                        site_type, n_bins
                    )

                    if len(bin_indices) > 0:
                        bin_indices = np.asarray(bin_indices)
                        auprc_values = np.asarray(auprc_values, dtype=float)

                        markers = ["o","s","D","^","v","<",">","P","X","*","h","H","+","x","d","1","2","3","4"]
                        marker = markers[model_index % len(markers)]
                        jitter = 0.08 * (np.random.rand(len(bin_indices)) - 0.5)

                        ax.scatter(
                            bin_indices + jitter,
                            auprc_values,
                            c=[color],
                            s=20, alpha=0.8,
                            marker=marker,
                            edgecolors="black",
                            linewidth=0.4,
                            label=model_tag
                        )

                        # Connecting lines
                        valid_mask = ~np.isnan(auprc_values)
                        if np.sum(valid_mask) > 1:
                            ax.plot(
                                bin_indices[valid_mask],
                                auprc_values[valid_mask],
                                color=color,
                                alpha=0.5,
                                linewidth=1
                            )

                        # ------------ NEW: Secondary Y-axis for median width -------------
                        ax2 = ax.twinx()
                        ax2.plot(
                            bin_indices, median_widths,
                            color="gray", linestyle="--", linewidth=1.5, alpha=0.8
                        )
                        ax2.set_ylabel("Median Width")
                        ax2.grid(False)
                        # -----------------------------------------------------------------


                
            # Only show legend in first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_tags = "_".join([tag for tag, _, _, _ in models_data])
        save_path = os.path.join(save_dir, f"auprc_scatter_by_fil_{model_tags}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved FIL-split scatter plot] {save_path}")
    
    plt.show()

# ---------------------------------------------------------
# Direct function to compute and plot AUPRC scatter from saved npz
# ---------------------------------------------------------

def plot_auprc_scatter_from_npz_files(npz_paths, model_tags, save_dir=None):
    """
    Load multiple npz files and plot combined scatter plots.
    
    Args:
        npz_paths: List of paths to npz files
        model_tags: List of model tags corresponding to each npz file
        save_dir: Directory to save plots
    """
    models_data = []
    
    for npz_path, model_tag in zip(npz_paths, model_tags):
        if not os.path.exists(npz_path):
            print(f"[Warning] NPZ file not found: {npz_path}")
            continue
        
        data = np.load(npz_path, allow_pickle=True)
        preds_all, labels_all, meta_all = data["preds"], data["labels"], data["meta"]
        
        models_data.append((model_tag, preds_all, labels_all, meta_all))
    
    if models_data:
        #plot_auprc_scatter(models_data, save_dir)
        plot_auprc_scatter_by_fil(models_data, save_dir)

# -------------------------
# Evaluation
# -------------------------

def run_model_prediction(df, save_dir, model, model_tag, fasta=None):
    """
    Run predictions for a given model, save .npz outputs.
    """
    # ----------------------------------
    # Setup
    # ----------------------------------
    os.makedirs(save_dir, exist_ok=True)
    
    FIL_order = ["Fi", "In", "La"]
    
    # Custom collate for single-exon processing
    def collate_fn_batch1(batch):
        return batch[0]
    
    is_spliceai = 'spliceai' in model_tag.lower()
    ds = SpliceDataset(df, fasta_obj=fasta, is_spliceai=is_spliceai)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn_batch1)
    
    all_preds, all_labels, all_meta = [], [], []
    
    # ----------------------------------
    # Setup device & inference mode
    # ----------------------------------
    if is_spliceai:
        # Configure TensorFlow GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[INFO] Using TensorFlow GPU: {gpus[0].name}")
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except Exception:
                pass
        else:
            print("[WARNING] No TensorFlow GPU detected. Running SpliceAI on CPU.")
    
    else:
        model.eval()
        model = model.to(DEVICE)
    
    # ----------------------------------
    # Inference loop
    # ----------------------------------
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            if batch_idx % 100 == 0:
                print(f"[Processing] Sample {batch_idx + 1}/{len(dl)}", flush=True)
            meta = batch["meta"]
            fil = meta.get("FIL", None)
            if fil not in FIL_order:
                continue
            
            # -----------------------------
            # Prediction
            # -----------------------------
            if is_spliceai:
                # Convert PyTorch tensor -> NumPy -> TF tensor
                seq_np = batch["seq"].unsqueeze(0).numpy().astype(np.float32)       # (1, 4, L)
                seq_tf = tf.convert_to_tensor(seq_np.transpose(0, 2, 1))            # (1, L, 4)
                # TF inference
                outputs_tf = model(seq_tf, training=False)                           # (1, L, 4)
                # Extract donor (2) and acceptor (1)
                preds = outputs_tf.numpy()[0, :, [2, 1]].T                          # (L_out, 2)
            else:
                seq = batch["seq"].unsqueeze(0).to(DEVICE)  # (1, 4, L_in)
                preds = model(seq).cpu().numpy()[0].T  # (L_out, 2)
            
            seq_len_out = preds.shape[0]
            exon_len = seq_len_out - 20  # 10bp flanking each side
            
            # -----------------------------
            # Construct labels (binary)
            # -----------------------------
            labels = np.zeros((seq_len_out, 2))
            labels[10, 1] = 1                    # 3'ss
            labels[10 + exon_len - 1, 0] = 1     # 5'ss
            
            # -----------------------------
            # Save predictions
            # -----------------------------
            all_preds.append(preds)
            all_labels.append(labels)
            all_meta.append(meta)
    
    # ----------------------------------
    # Save NPZ outputs
    # ----------------------------------
    npz_path = os.path.join(save_dir, f"{model_tag}_pred_label_pairs.npz")
    np.savez_compressed(
        npz_path,
        preds=np.array(all_preds, dtype=object),
        labels=np.array(all_labels, dtype=object),
        meta=np.array(all_meta, dtype=object),
    )
    print(f"[Saved predictions] {npz_path}")
    
    return npz_path

# -------------------------
# Main
# -------------------------

def main(data_path, out_dir, model_configs, ref_file=None, force_rerun=False):
    """
    Main workflow for generating AUPRC scatter plots for multiple models.
    
    Args:
        data_path: Path to exon dataset TSV
        out_dir: Output directory for results
        model_configs: List of tuples (model_tag, model_path, model_type)
                       model_type can be 'simple_2layer', 'simple_3layer', 'simple_sse', 'spliceai', or None for auto-detect
        ref_file: Path to reference fasta (required for spliceai models)
        force_rerun: If True, rerun predictions even if npz files exist
    """
    # ----------------------------------
    # Setup directories
    # ----------------------------------
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(data_path, sep="\t")
    print(f"[Loaded dataset] {data_path} ({len(df)} rows)")
    
    # ----------------------------------
    # Process each model
    # ----------------------------------
    npz_paths = []
    model_tags = []
    
    for model_tag, model_path, model_type in model_configs:
        npz_path = os.path.join(out_dir, f"{model_tag}_pred_label_pairs.npz")
        
        if os.path.exists(npz_path) and not force_rerun:
            print(f"[Reusing predictions for {model_tag}] {npz_path}")
            npz_paths.append(npz_path)
            model_tags.append(model_tag)
            continue
        
        # Load and run model
        model, loaded_model_type, _ = load_model_by_tag(model_tag, model_path, model_type)
        if model is None:
            print(f"[Skipping] Could not load model: {model_tag}")
            continue
        
        if loaded_model_type == 'spliceai':
            if not ref_file:
                raise ValueError("Reference genome path must be provided for SpliceAI.")
            fasta = pyfastx.Fasta(ref_file)
            npz_path = run_model_prediction(df, out_dir, model, model_tag, fasta=fasta)
        else:
            npz_path = run_model_prediction(df, out_dir, model, model_tag)
        
        npz_paths.append(npz_path)
        model_tags.append(model_tag)
    
    # ----------------------------------
    # Plot combined scatter plots
    # ----------------------------------
    if npz_paths:
        print("\n=== Generating combined scatter plots ===")
        plot_auprc_scatter_from_npz_files(npz_paths, model_tags, out_dir)
    
    # ----------------------------------
    # Merge all predictions → Summary TSV
    # ----------------------------------
    print("\n=== Generating unified prediction summary ===")
    summary_path = os.path.join(out_dir, "merged_prediction_summary.tsv")
    generate_prediction_summary(df, out_dir, summary_path)
    
    print("\nAll analyses completed successfully.")
    print(f"Results and summary saved in: {out_dir}")

# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate multiple splice site prediction models and plot AUPRC scatter plots"
    )
    parser.add_argument("--data", required=True, help="Path to exon dataset TSV")
    parser.add_argument("--out_dir", required=True, help="Output directory for results")
    parser.add_argument("--reference", default=None, help="Path to reference fasta (required for spliceai models)")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun predictions even if npz files exist")
    
    # Model configuration arguments
    parser.add_argument("--models", nargs='+', action='append', 
                       help="Model configuration: tag:path:type (type can be 2layer, 3layer, sse, spliceai, or auto)")
    
    args = parser.parse_args()
    
    # Parse model configurations
    model_configs = []
    if args.models:
        for model_spec in args.models[0]:
            parts = model_spec.split(':')
            if len(parts) == 2:
                tag, path = parts
                model_type = None  # Auto-detect
            elif len(parts) == 3:
                tag, path, model_type = parts
                # Convert type string to standardized format
                if model_type.lower() == 'auto':
                    model_type = None
                elif model_type.lower() in ['2layer', 'simple_2layer']:
                    model_type = 'simple_2layer'
                elif model_type.lower() in ['3layer', 'simple_3layer', 'triple']:
                    model_type = 'simple_3layer'
                elif model_type.lower() in ['binary', 'simple_binary']:
                    model_type = 'simple_binary'
                elif model_type.lower() == 'spliceai':
                    model_type = 'spliceai'
                else:
                    print(f"[Warning] Unknown model type: {model_type}. Using auto-detect.")
                    model_type = None
            else:
                raise ValueError(f"Invalid model specification: {model_spec}. Use tag:path or tag:path:type")
            
            model_configs.append((tag, path, model_type))
    
    if not model_configs:
        print("[Warning] No models specified. Use --models to specify models.")
        print("Example: --models simple_2layer:./models/simple_2layer.pth:2layer \\")
        print("         --models simple_3layer:./models/simple_3layer.pth:3layer \\")
        print("         --models spliceai:./models/spliceai.h5:spliceai")
        exit(1)
    
    print(f"[Info] Processing {len(model_configs)} models:")
    for tag, path, mtype in model_configs:
        print(f"  - {tag}: {path} ({mtype if mtype else 'auto-detect'})")
    
    main(args.data, args.out_dir, model_configs, args.reference, args.force_rerun)