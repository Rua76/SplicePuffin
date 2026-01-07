#!/usr/bin/env python3
"""
evaluate_connor_testset.py

Evaluate splice site prediction model on exon dataset with sub-category breakdown.
"""

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns

from models import SimpleNetModified_DA_SSE, SimpleNetModified_DA, SimpleNetModified_DA_TripleLayers
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
def load_model_by_tag(model_tag, SSE_path, binary_path, spliceai_path):
    """
    Load the appropriate model based on tag and directory structure.
    Returns (model_object, model_type, save_path)
    """
    if model_tag == "simple_SSE":
        model_path =SSE_path
        model = SimpleNetModified_DA_SSE(input_channels=4)
        model = model.to(DEVICE)
    elif model_tag == "simple_Binary":
        model_path = binary_path
        model = SimpleNetModified_DA(input_channels=4)
        model = model.to(DEVICE)
    elif model_tag == "spliceai":
        model_path = spliceai_path
        model = load_model(model_path, compile=False)
    else:
        raise ValueError(f"Unknown model tag: {model_tag}")

    if not os.path.exists(model_path):
        print(f"[Warning] Model checkpoint not found for {model_tag}: {model_path}")
        return None, None, None

    print(f"[Loading model] {model_path}")
    if model_tag != "spliceai":
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
    return model, model_tag, model_path

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
        
        # Use the correct column names tht match the original dataframe
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
# 1. Compute per-bin AUPRC values
# ---------------------------------------------------------
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
# 2. Convert auprc_maps to DataFrames (heatmaps)
# ---------------------------------------------------------
def compute_heatmaps_from_auprc_maps(auprc_maps, FIL_order):
    def sort_bin_labels(labels, prefix):
        try:
            return sorted(labels, key=lambda x: float(x.replace(prefix, "")))
        except Exception:
            return sorted(labels)

    heatmaps = {}
    for fil in FIL_order:
        heatmaps[fil] = {}
        for site in ["5ss", "3ss"]:
            keys = list(auprc_maps[fil][site].keys())
            Wi_bins = sort_bin_labels(set(k[0] for k in keys), prefix="Wi")
            GC_bins = sort_bin_labels(set(k[1] for k in keys), prefix="GC")

            hm = pd.DataFrame(index=Wi_bins, columns=GC_bins, data=np.nan)
            for k in keys:
                y_true = auprc_maps[fil][site][k]['y_true']
                y_score = auprc_maps[fil][site][k]['y_score']
                if sum(y_true) > 0:
                    precision, recall, _ = precision_recall_curve(y_true, y_score)
                    hm.loc[k[0], k[1]] = auc(recall, precision)
            heatmaps[fil][site] = hm.astype(float)

    return heatmaps


# ---------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------
def plot_auprc_heatmap_grid(heatmaps, FIL_order, FIL_labels, model_tag="Model", save_dir=None):
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    z_min, z_max = -3, 3

    # -----------------------------------------------------
    # z-score scaling PER heatmap (per FIL × site), not global
    # -----------------------------------------------------
    zscore_maps = {}
    for fil in FIL_order:
        zscore_maps[fil] = {}
        for site, hm in heatmaps[fil].items():
            vals = hm.values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 1:
                zhm = (hm - np.nanmean(vals)) / np.nanstd(vals)
            else:
                zhm = hm * 0
            zscore_maps[fil][site] = zhm

    # -----------------------------------------------------
    # Plot layout
    # -----------------------------------------------------
    fig = plt.figure(figsize=(12, 14))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 0.05], wspace=0.2, hspace=0.35)

    for i, fil in enumerate(FIL_order):
        for j, site in enumerate(["5ss", "3ss"]):
            ax = fig.add_subplot(gs[i, j])
            hm, zhm = heatmaps[fil][site], zscore_maps[fil][site]
            sns.heatmap(
                zhm, cmap=cmap, annot=hm, fmt=".2f", ax=ax,
                vmin=z_min, vmax=z_max, cbar=False, annot_kws={"size": 8}
            )
            ax.set_title(f"{site.upper()} — {FIL_labels[fil]}")
            ax.set_xlabel("GC_bin")
            ax.set_ylabel("Wi_bin")
            ax.invert_yaxis()

    # Shared colorbar
    ax_cbar = fig.add_subplot(gs[:, 2])
    norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=ax_cbar).set_label("Z-score of AUPRC")

    plt.suptitle(f"{model_tag} — AUPRC by FIL, WI × GC (per-grid z-score)", fontsize=16)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_tag}_auprc_heatmaps.pdf")
        plt.savefig(path, dpi=300, bbox_inches="tight", format="pdf")
        print(f"[Saved plot] {path}")
    plt.close()


# ---------------------------------------------------------
# Direct function to compute and plot AUPRC heatmaps from saved npz
# ---------------------------------------------------------
def plot_auprc_heatmaps_from_npz(npz_path, save_dir=None, model_tag=None):
    data = np.load(npz_path, allow_pickle=True)
    preds_all, labels_all, meta_all = data["preds"], data["labels"], data["meta"]

    FIL_order = ["Fi", "In", "La"]
    FIL_labels = {"Fi": "First Exon", "In": "Intermediate Exon", "La": "Last Exon"}

    auprc_maps = compute_auprc_maps(preds_all, labels_all, meta_all, FIL_order)
    heatmaps = compute_heatmaps_from_auprc_maps(auprc_maps, FIL_order)
    plot_auprc_heatmap_grid(heatmaps, FIL_order, FIL_labels, model_tag, save_dir)


# -------------------------
# Evaluation
# -------------------------
def run_model_prediction(df, save_dir, model, model_tag, fasta=None):
    """
    Run predictions for a given model (simple_SSE, simple_Binary, or spliceai),
    save .npz outputs, and generate AUPRC heatmaps.

    Handles PyTorch (simple_*) and TensorFlow (spliceai) models efficiently.
    """
    # ----------------------------------
    # Setup
    # ----------------------------------
    os.makedirs(save_dir, exist_ok=True)

    FIL_order = ["Fi", "In", "La"]
    FIL_labels = {"Fi": "First Exon", "In": "Intermediate Exon", "La": "Last Exon"}

    # Custom collate for single-exon processing
    def collate_fn_batch1(batch):
        return batch[0]

    is_spliceai = model_tag.lower() == "spliceai"
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

    # ----------------------------------
    # Plot AUPRC heatmaps
    # ----------------------------------
    print(f"[Computing AUPRC heatmaps for {model_tag}]")
    auprc_maps = compute_auprc_maps(all_preds, all_labels, all_meta, FIL_order)
    heatmaps = compute_heatmaps_from_auprc_maps(auprc_maps, FIL_order)
    plot_auprc_heatmap_grid(heatmaps, FIL_order, FIL_labels, model_tag.upper(), save_dir)


# -------------------------
# Plotting functions
# -------------------------

def plot_site_counts(df, save_dir=None):
    """Plot 5'SS and 3'SS site counts by GC_bin × Wi_bin, split by FIL (Fi, In, La), with a standalone colorbar."""


    # Parse code field into structured columns
    meta_df = df['code'].apply(parse_code).apply(pd.Series)
    df = pd.concat([df, meta_df], axis=1).reset_index(drop=True)

    df['has_5ss'] = df['five_SS_position'].notna()
    df['has_3ss'] = df['three_SS_position'].notna()

    def sort_bins(df_):
        try:
            df_ = df_.loc[sorted(df_.index, key=lambda x: float(x))]
            df_ = df_[sorted(df_.columns, key=lambda x: float(x))]
        except Exception:
            pass
        return df_

    FIL_order = ["Fi", "In", "La"]
    FIL_labels = {"Fi": "First Exon", "In": "Intermediate Exon", "La": "Last Exon"}

    heatmaps = {}
    for fil in FIL_order:
        sub = df.loc[df["FIL"].values == fil].copy()
        if sub.empty:
            print(f"[Warning] No data for FIL='{fil}'")
            continue

        five_counts = (
            sub[sub["has_5ss"]]
            .groupby(["Wi_bin", "GC_bin"])
            .size()
            .unstack(fill_value=0)
        )
        three_counts = (
            sub[sub["has_3ss"]]
            .groupby(["Wi_bin", "GC_bin"])
            .size()
            .unstack(fill_value=0)
        )

        heatmaps[fil] = (sort_bins(five_counts), sort_bins(three_counts))

    # Shared color scale
    all_max = 0
    for five, three in heatmaps.values():
        all_max = max(all_max, five.values.max() if not five.empty else 0)
        all_max = max(all_max, three.values.max() if not three.empty else 0)

    vmin, vmax = 0, all_max
    cmap = sns.light_palette("blue", as_cmap=True)

    # ---------------------------
    # Create GridSpec: 3 rows x 3 columns (last col for colorbar)
    # ---------------------------
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(nrows=3, ncols=3, width_ratios=[1, 1, 0.05], wspace=0.3, hspace=0.4)

    for i, fil in enumerate(FIL_order):
        if fil not in heatmaps:
            continue
        five_counts, three_counts = heatmaps[fil]

        ax5 = fig.add_subplot(gs[i, 0])
        sns.heatmap(
            five_counts,
            cmap=cmap,
            annot=True,
            fmt="d",
            ax=ax5,
            vmin=vmin,
            vmax=vmax,
            cbar=False
        )
        ax5.set_title(f"5′SS — {FIL_labels[fil]}")
        ax5.set_xlabel("GC_bin")
        ax5.set_ylabel("Wi_bin")
        ax5.invert_yaxis()  # Flip so Wi1 is at bottom

        ax3 = fig.add_subplot(gs[i, 1])
        sns.heatmap(
            three_counts,
            cmap=cmap,
            annot=True,
            fmt="d",
            ax=ax3,
            vmin=vmin,
            vmax=vmax,
            cbar=False
        )
        ax3.set_title(f"3′SS — {FIL_labels[fil]}")
        ax3.set_xlabel("GC_bin")
        ax3.set_ylabel("Wi_bin")
        ax3.invert_yaxis()  # Flip so Wi1 is at bottom

    # Standalone colorbar in the last column
    ax_cbar = fig.add_subplot(gs[:, 2])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label("Site count")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        outpath = os.path.join(save_dir, "splice_site_counts_by_FIL.pdf")
        plt.savefig(outpath, dpi=300, format='pdf', bbox_inches='tight')
        print(f"[Saved] {outpath}")
    plt.close()


# -------------------------
# Main
# -------------------------

def main(data_path, out_dir, simple_SSE_path, simple_Binary_path,spliceai_path, ref_file=None):
    """
    Main workflow for generating AUPRC heatmaps and site distributions
    for both 'simple' and 'spliceai' models.

    Automatically detects existing prediction files and reuses them
    to avoid redundant model evaluation.
    """
    # ----------------------------------
    # Setup directories
    # ----------------------------------
    fig_dir = os.path.join('./', out_dir)
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(data_path, sep="\t")
    print(f"[Loaded dataset] {data_path} ({len(df)} rows)")

    # ----------------------------------
    # Step 1: Plot general site distributions
    # ----------------------------------
    plot_site_counts(df, fig_dir)

    # ----------------------------------
    # Step 2: Model evaluation or plotting for each type
    # ----------------------------------
    for model_tag in ["simple_SSE", "simple_Binary", "spliceai"]:
        npz_path = os.path.join(fig_dir, f"{model_tag}_pred_label_pairs.npz")

        if os.path.exists(npz_path):
            print(f"[Reusing predictions for {model_tag}] {npz_path}")
            plot_auprc_heatmaps_from_npz(npz_path, fig_dir, model_tag.upper())
            continue

        model, _, model_path = load_model_by_tag(model_tag, simple_SSE_path, simple_Binary_path,spliceai_path)
        if model is None:
            continue

        if model_tag == "spliceai":
            if not ref_file:
                raise ValueError("Reference genome path must be provided for SpliceAI.")
            fasta = pyfastx.Fasta(ref_file)
            run_model_prediction(df, fig_dir, model, model_tag="spliceai", fasta=fasta)
        else:
            run_model_prediction(df, fig_dir, model, model_tag=model_tag)

    # ----------------------------------
    # Step 3: Merge all predictions → Summary TSV
    # ----------------------------------
    print("\n=== Generating unified prediction summary ===")
    summary_path = os.path.join(fig_dir, "merged_prediction_summary.tsv")
    generate_prediction_summary(df, fig_dir, summary_path)

    print("\nAll analyses completed successfully.")
    print(f"Results and summary saved in: {fig_dir}")

# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to exon dataset TSV")
    parser.add_argument("--save_dir", help="Directory containing model checkpoints")
    parser.add_argument("--out_dir", help="Directory for model outputs and figures") 
    parser.add_argument("--simple_SSE_path",  help="path to simple_SSE model directory")
    parser.add_argument("--simple_Binary_path", help="path to simple_Binary model directory")
    parser.add_argument("--spliceai_path", help="path to SpliceAI model directory")
    parser.add_argument("--reference_file", default=None, help="Path to reference fasta (required if with_ref_genome)")
    args = parser.parse_args()

    main(args.data, args.out_dir, args.simple_SSE_path, args.simple_Binary_path,args.spliceai_path, args.reference_file)
