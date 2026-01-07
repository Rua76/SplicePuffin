import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from sklearn.metrics import precision_recall_curve, auc
from models import SimpleNetModified_DA_SSE, H5Dataset, SimpleNetModified_DA_TripleLayers
import functools

# Lightweight imports only
import pyfastx  # Keep this as it's relatively lightweight

from SplicingStats.MaxEnt import MaxEnt
from scipy.special import expit   # stable sigmoid

from utils import one_hot_encode_sequence, one_hot_to_sequence

ME = MaxEnt()

def evaluate_model_auprc_MaxEntScan(test_dl, return_curves=False, verbose=True, debug=True):
    """
    Evaluate MaxEntScan on test_dl and compute auPRC for Donor and Acceptor channels.
    With optional debug output to inspect window/score assignment logic.
    """

    all_outputs, all_targets, all_masks = [], [], []

    if verbose:
        print("\n=== Evaluating MaxEntScan on Test Set ===")

    seq_counter = 0  # for limiting debug prints

    for batch_idx, batch in enumerate(test_dl):
        if len(batch) != 5:
            raise ValueError("Each element of test_dl must be (inputs, targets, M, coord, batch_seq)")
        inputs, targets, M, coord, batch_seq = batch

        if batch_idx % 50 == 0 and verbose:
            print(f"Processing batch {batch_idx}/{len(test_dl)}")

        batch_size = len(batch_seq)
        seq_len = len(batch_seq[0])
        outputs_batch = np.full((batch_size, 2, seq_len), -999.0, dtype=np.float32)

        for i, full_seq in enumerate(batch_seq):
            s = full_seq.upper()
            L = len(s)
            donor_scores = np.full(L, np.nan, dtype=np.float32)
            acceptor_scores = np.full(L, np.nan, dtype=np.float32)

            if debug and seq_counter < 2:
                print(f"\n[DEBUG] Processing sequence {seq_counter} (len={L})")

            # === DONOR (5'SS) ===
            if L >= 9:
                windows9 = [s[w:w+9] for w in range(L - 8)]
                scores9 = np.array(ME.compute_score(windows9, 0), dtype=np.float32)

                # Assign to the last exon base (index +2)
                assign_positions = np.arange(len(scores9)) + 2
                valid_mask = assign_positions < L
                donor_scores[assign_positions[valid_mask]] = scores9[valid_mask]

                if debug and seq_counter < 2:
                    print(f"  Donor: generated {len(windows9)} windows of 9nt")
                    for w in range(0, min(3, len(windows9))):
                        ap = assign_positions[w]
                        print(f"    w={w:4d} | assign→pos={ap} | window={windows9[w]} | score={scores9[w]:.3f}")

            # === ACCEPTOR (3'SS) ===
            if L >= 23:
                windows23 = [s[w:w+23] for w in range(L - 22)]
                scores23 = np.array(ME.compute_score(windows23, 0), dtype=np.float32)
                assign_positions = np.arange(len(scores23)) + 20
                valid_mask = assign_positions < L
                acceptor_scores[assign_positions[valid_mask]] = scores23[valid_mask]

                if debug and seq_counter < 2:
                    print(f"  Acceptor: generated {len(windows23)} windows of 23nt")
                    for w in range(0, min(3, len(windows23))):
                        ap = assign_positions[w]
                        print(f"    w={w:4d} | assign→pos={ap} | window={windows23[w]} | score={scores23[w]:.3f}")

            # Replace NaNs with very negative scores
            donor_scores_nonan = np.where(np.isfinite(donor_scores), donor_scores, -100.0)
            acceptor_scores_nonan = np.where(np.isfinite(acceptor_scores), acceptor_scores, -100.0)

            outputs_batch[i, 0, :] = donor_scores_nonan
            outputs_batch[i, 1, :] = acceptor_scores_nonan

            if debug and seq_counter < 2:
                # Print small region of the sequence and assigned scores
                center = L // 2
                print("  --- Sample donor scores (center ±10) ---")
                print(donor_scores_nonan[center-10:center+10])
                print("  --- Sample acceptor scores (center ±10) ---")
                print(acceptor_scores_nonan[center-10:center+10])

            seq_counter += 1

        # Convert log-odds → probabilities
        outputs_prob = expit(outputs_batch)
        all_outputs.append(outputs_prob)

        if hasattr(targets, "cpu"):
            all_targets.append(targets.cpu().numpy())
        else:
            all_targets.append(np.asarray(targets))

        if hasattr(M, "cpu"):
            all_masks.append(M.cpu().numpy())
        else:
            all_masks.append(np.asarray(M))

    # --- concatenate all batches ---
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    seq_len = all_outputs.shape[2]

    # Trim 500bp padding from each side
    start, end = 500, seq_len - 500
    all_outputs = all_outputs[:, :, start:end]


    # --- Compute AUPRC ---
    auprc_results, curves = {}, {}

    for ch_idx, ch_name in enumerate(["Donor", "Acceptor"]):
        y_scores = all_outputs[:, ch_idx, :]
        y_true = all_targets[:, ch_idx, :]
        valid = all_masks == 1

        y_scores_valid = y_scores[valid].flatten()
        y_true_valid = y_true[valid].flatten()

        if len(np.unique(y_true_valid)) < 2:
            auprc_results[ch_name] = 0.0
            continue

        precision, recall, _ = precision_recall_curve(y_true_valid, y_scores_valid)
        auprc_results[ch_name] = auc(recall, precision)
        if return_curves:
            curves[ch_name] = (precision, recall)

    # Combined (micro-average)
    combined_scores, combined_labels = [], []
    for ch in range(2):
        y_scores = all_outputs[:, ch, :]
        y_true = all_targets[:, ch, :]
        valid = all_masks == 1
        combined_scores.append(y_scores[valid])
        combined_labels.append(y_true[valid])
    y_scores_comb = np.concatenate(combined_scores)
    y_true_comb = np.concatenate(combined_labels)

    if len(np.unique(y_true_comb)) >= 2:
        precision_c, recall_c, _ = precision_recall_curve(y_true_comb, y_scores_comb)
        auprc_results["Combined"] = auc(recall_c, precision_c)
        if return_curves:
            curves["Combined"] = (precision_c, recall_c)
    else:
        auprc_results["Combined"] = 0.0

    if return_curves:
        return auprc_results, curves
    else:
        return auprc_results

# -------------------------------
# Evaluate model (supports both output formats)
# -------------------------------
def evaluate_model_auprc_DLmodels(model, test_dl, replicate_num, return_curves=False, is_spliceai=False, model_name=""):
    """
    Evaluate model and compute auPRC for Donor and Acceptor channels.
    Compatible with:
      - Custom model: output shape (N, 2, 4000)
      - SpliceAI: output shape (N, seq_len, 3)
    """
    
    all_outputs, all_targets, all_masks = [], [], []
    if is_spliceai:
        print(f"\n=== Evaluating {model_name} on Test Set ===")
    else:
        model.eval()
        print(f"\n=== Evaluating Replicate {replicate_num} on Test Set ===")

    with torch.no_grad():
        for batch_idx, (inputs, targets, M, coord) in enumerate(test_dl):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(test_dl)}")
            # unify output to shape (N, 2, 4000)
            if is_spliceai:
                # Convert PyTorch tensor to NumPy and adjust dimensions for SpliceAI
                # SpliceAI expects input shape: (batch_size, sequence_length, 4)
                inputs_np = inputs.numpy().astype(np.float32).transpose(0, 2, 1)   # Convert to NumPy and (N, 5000, 4)            
                # Get predictions from SpliceAI
                outputs_np = model.predict(inputs_np)  # Shape: (N, 5000, 3)
                # Extract acceptor (index 1) and donor (index 2) → (N, 2, 5000)
                outputs_np = outputs_np[:, :, [2, 1]].transpose(0, 2, 1)
                # Remove 500bp padding from both ends → (N, 2, 4000)
                outputs_np = outputs_np[:, :, 500:-500]                
                # Convert back to PyTorch tensor for consistency
                all_outputs.append(outputs_np)
                all_targets.append(targets.numpy())
                all_masks.append(M.numpy())
            else:
                inputs, targets, M = inputs.cuda(), targets.cuda(), M.cuda()
                # custom model outputs (N, 2, 4000)
                outputs = model(inputs)

                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_masks.append(M.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)  # (N, 2, 4000)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    auprc_results = {}
    curves = {}

    for channel_idx, channel_name in enumerate(['Donor', 'Acceptor']):
        y_scores = all_outputs[:, channel_idx, :]
        y_true = all_targets[:, channel_idx, :]
        valid_positions = all_masks == 1

        y_scores_valid = y_scores[valid_positions].flatten()
        y_true_valid = y_true[valid_positions].flatten()

        if len(np.unique(y_true_valid)) < 2:
            auprc_results[channel_name] = 0.0
            continue

        precision, recall, _ = precision_recall_curve(y_true_valid, y_scores_valid)
        auprc = auc(recall, precision)
        auprc_results[channel_name] = auprc
        if return_curves:
            curves[channel_name] = (precision, recall)

    # Combined (micro-average)
    combined_scores, combined_labels = [], []
    for ch in range(2):
        y_scores = all_outputs[:, ch, :]
        y_true = all_targets[:, ch, :]
        valid = all_masks == 1
        combined_scores.append(y_scores[valid])
        combined_labels.append(y_true[valid])
    y_scores_combined = np.concatenate(combined_scores)
    y_true_combined = np.concatenate(combined_labels)

    if len(np.unique(y_true_combined)) >= 2:
        precision_c, recall_c, _ = precision_recall_curve(y_true_combined, y_scores_combined)
        auprc_c = auc(recall_c, precision_c)
        auprc_results["Combined"] = auprc_c
        if return_curves:
            curves["Combined"] = (precision_c, recall_c)
    else:
        auprc_results["Combined"] = 0.0

    if return_curves:
        return auprc_results, curves
    else:
        return auprc_results


# -------------------------------
# Helper: Prepare SpliceAI input
# -------------------------------
def prepare_input_for_spliceai(one_hot_input, model_context):
    """
    pad with N's (0,0,0,0) according to model context window.
    """
    x_core = one_hot_input  # (B, 4, 5000)
    pad_len = model_context // 2
    pad = torch.zeros((x_core.shape[0], 4, pad_len), dtype=x_core.dtype)
    return torch.cat([pad, x_core, pad], dim=-1)  # (B, 4, 5000 + context)



def prepare_input_with_reference(one_hot_input, coord, fasta, model_context):
    """
    Pad each input with upstream/downstream reference sequence using FASTA.
    
    Args:
        one_hot_input: Tensor (B, 4, L_core)
        reference_fasta_path: path to reference fasta
        model_context: total context window (pad_len = model_context // 2)
    
    Returns:
        Tensor (B, 4, L_core + model_context)
    """
    pad_len = model_context // 2
    x_core = one_hot_input
    batch_size, _, L_core = x_core.shape
    padded_inputs = []

    chroms, starts, ends, strands = coord
    for i in range(batch_size):
        # pull sample-specific chrom/start/end/strand from parallel lists
        chrom = chroms[i].decode() if isinstance(chroms[i], bytes) else chroms[i]
        strand = strands[i].decode() if isinstance(strands[i], bytes) else strands[i]
        # starts/ends may be bytes/strings/numbers
        start = int(starts[i])
        end = int(ends[i])
    # Ensure start < end for slicing
        s, e = (start, end) if start < end else (end, start)

        # Extract upstream and downstream positions
        if strand == '+':
            upstream_start = max(0, s - pad_len)
            upstream_end = s
            downstream_start = e
            downstream_end = e + pad_len
        else:
            # negative strand
            upstream_start = e
            upstream_end = e + pad_len
            downstream_start = max(0, s - pad_len)
            downstream_end = s

        # Fetch sequences from reference
        if strand == '+':
            upstream_seq = fasta[chrom][upstream_start:upstream_end].seq
            downstream_seq = fasta[chrom][downstream_start:downstream_end].seq
        else:
            # Use .antisense for negative strand
            upstream_seq = fasta[chrom][upstream_start:upstream_end].antisense
            downstream_seq = fasta[chrom][downstream_start:downstream_end].antisense

        # Core sequence from one-hot
        core_seq = one_hot_to_sequence(x_core[i].numpy())

        # One-hot encode sequences
        x_up = one_hot_encode_sequence(upstream_seq)
        x_core_encoded = one_hot_encode_sequence(core_seq)
        x_down = one_hot_encode_sequence(downstream_seq)
        assert x_up.shape[1] + x_core_encoded.shape[1] + x_down.shape[1] == L_core + model_context, \
            f"Length mismatch: {x_up.shape[1]} + {x_core_encoded.shape[1]} + {x_down.shape[1]} != {L_core + model_context}"

        # Concatenate along length axis
        x_padded = np.concatenate([x_up, x_core_encoded, x_down], axis=1)
        padded_inputs.append(x_padded)

    return torch.from_numpy(np.stack(padded_inputs, axis=0)).float()  # (B, 4, L_core + context)

# -------------------------------
# Plot PR curves
# -------------------------------
def plot_pr_curves(curves_dict, filename):
    channels = ['Donor', 'Acceptor', 'Combined']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, ch in enumerate(channels):
        ax = axes[i]
        for model_name, curve_data in curves_dict.items():
            if ch in curve_data:
                precision, recall = curve_data[ch]
                auprc = auc(recall, precision)
                ax.plot(
                    recall, precision, lw=2,
                    label=f"{model_name} (AUPRC={auprc:.3f})"
                )

        ax.set_title(f"{ch} Precision-Recall Curve", fontsize=14)
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        # Smaller legend in top-right corner
        ax.legend(
            loc='upper right',
            fontsize=9,
            frameon=True,
            fancybox=True,
            framealpha=0.8
        )

    fig.suptitle("Model Performance Comparison: Precision-Recall Curves", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300, format='pdf')

def debug_dataset_vs_reference(test_dl, fasta, context=10000, max_batches=5, show_padding=False):
    """
    Debug function: check that the one-hot input core sequence matches the reference genome
    sequence at the exact start:end coordinates for each sample in the batch.

    Args:
        test_dl: DataLoader for test dataset (yields inputs, targets, M, coord)
                 where coord is [chroms_list, starts_list, ends_list, strands_list]
        fasta: pyfastx.Fasta object (preloaded)
        context: (unused here except for optional padding checks) total context window
        max_batches: how many batches to check
        show_padding: if True, also extract & compare padding regions (upstream/downstream)
    """
    pad_len = context // 2

    def one_hot_to_seq_batch(x):
        """Convert (B,4,L) numpy array or torch tensor to list of length-L DNA strings."""
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        seqs = []
        for b in range(x.shape[0]):
            # x[b] shape (4, L)
            col_sums = x[b].sum(axis=0)
            indices = x[b].argmax(axis=0)
            seq_chars = []
            for idx, s in zip(indices, col_sums):
                if s == 0:
                    seq_chars.append("N")
                else:
                    seq_chars.append("ACGT"[int(idx)])
            seqs.append("".join(seq_chars))
        return seqs

    for batch_idx, (inputs, targets, M, coord) in enumerate(test_dl):
        if batch_idx >= max_batches:
            break

        # inputs expected shape (B,4,L)
        input_seqs = one_hot_to_seq_batch(inputs)
        batch_size = len(input_seqs)

        # coord expected as 4 lists: [chroms, starts, ends, strands]
        if not (isinstance(coord, (list, tuple)) and len(coord) == 4):
            raise ValueError("coord must be a 4-element list/tuple: [chroms, starts, ends, strands]")

        chroms, starts, ends, strands = coord

        for i in range(batch_size):
            # pull sample-specific chrom/start/end/strand from parallel lists
            chrom = chroms[i].decode() if isinstance(chroms[i], bytes) else chroms[i]
            strand = strands[i].decode() if isinstance(strands[i], bytes) else strands[i]
            # starts/ends may be bytes/strings/numbers
            s = int(starts[i])
            e = int(ends[i])

            # Ensure s <= e for slicing pyfastx
            start, end = (s, e) if s <= e else (e, s)

            # Extract reference core sequence at exact coordinates
            # For +: use .seq of slice; for -: use .antisense of same slice
            ref_region = fasta[chrom][start:end]
            if strand == "+":
                ref_core = ref_region.seq
            else:
                # antisense corresponds to reverse-complement of genomic slice
                ref_core = ref_region.antisense

            input_core = input_seqs[i]

            # If lengths mismatch, report both lengths and compare up to min length
            len_in = len(input_core)
            len_ref = len(ref_core)
            min_len = min(len_in, len_ref)

            # compute match/mismatch on the overlapping region
            match_count = sum(1 for a, b in zip(input_core[:min_len], ref_core[:min_len]) if a == b)
            mismatch_count = min_len - match_count

            print(f"[Batch {batch_idx}, Sample {i}] {chrom}:{s}-{e} ({strand})")
            print(f"  Input core len: {len_in}, Ref core len: {len_ref}")
            print(f"  Overlap len: {min_len}; Matches: {match_count}; Mismatches: {mismatch_count}")

            if len_in != len_ref:
                print("  NOTE: length mismatch between input and reference core sequence.")

            if mismatch_count > 0:
                # show first 100 chars for diagnosis
                show_n = 100
                print("  Input core (first 100):", input_core[:show_n])
                print("  Ref   core (first 100):", ref_core[:show_n])

            # Optional: also check that surrounding padding (if present in input) matches reference flanks
            if show_padding:
                # upstream and downstream coordinates for padding
                up_start = max(0, start - pad_len)
                up_end = start
                down_start = end
                down_end = end + pad_len

                # get sequences (handle negative strand: antisense)
                up_slice = fasta[chrom][up_start:up_end]
                down_slice = fasta[chrom][down_start:down_end]
                if strand == "+":
                    ref_up = up_slice.seq
                    ref_down = down_slice.seq
                else:
                    # for - strand, upstream (genomic upstream) becomes antisense of downstream slice, etc.
                    ref_up = down_slice.antisense  # because genomic downstream corresponds to upstream on - strand
                    ref_down = up_slice.antisense

                # if your input contains padding concatenated to core, you could compare them here
                print(f"  Padding check (len target pad_len={pad_len}): up={len(ref_up)}, down={len(ref_down)}")

        print("-" * 60)

# Cache for SpliceAI models to avoid reloading
@functools.lru_cache(maxsize=1)
def load_spliceai_model_cached(model_path='../../SpliceAI/spliceai/models/spliceai1.h5'):
    """Cache loaded SpliceAI model to avoid repeated TensorFlow initialization"""
    print(f"Loading SpliceAI model from {model_path}...")
    from tensorflow.keras.models import load_model
    return load_model(model_path, compile=False)

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Evaluate replicate models by auPRC.")
    parser.add_argument("--test_data", required=True, help="Path to test dataset (.h5)")
    parser.add_argument("--save_dir", required=True, help="Directory with saved model.rep*.pth files")
    parser.add_argument("--num_replicates", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot", action="store_true", default=True, help="Generate auPRC plot (default: True)")
    parser.add_argument("--mode", choices=["normal", "with_ref_genome", 'debug'], default="normal",
                        help="Evaluation mode: normal (no genome) or with_ref_genome")
    parser.add_argument("--reference_file", default=None, help="Path to reference fasta (required if with_ref_genome)")
    parser.add_argument("--context", type=int, default=10000, help="Context window for SpliceAI-like padding")
    parser.add_argument("--output_file", default="model_auprc_comparison.pdf", help="File name for PR curve plot")
    parser.add_argument("--skip_spliceai", action="store_true", 
                        help="Skip SpliceAI evaluation to speed up")
    parser.add_argument("--skip_maxent", action="store_true",
                        help="Skip MaxEntScan evaluation to speed up")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()

    print(f"Script initialization time: {time.time() - start_time:.2f}s")
    
    # --- Load test dataset (lightweight) ---
    t0 = time.time()
    print(f"\nLoading test data from {args.test_data}")
    test_dataset = H5Dataset(args.test_data)
    test_dl = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"Dataset loaded in {time.time() - t0:.2f}s")
    
    results, curves = {}, {}
    
    # --- Debug mode: check dataset vs reference genome ---
    if args.mode == "debug":
        if not args.reference_file:
            raise ValueError("Reference file must be provided in debug mode")
        print(f"\n=== DEBUG MODE: Checking input vs reference genome ===")
        t1 = time.time()
        fasta = pyfastx.Fasta(args.reference_file)
        print(f"FASTA loaded in {time.time() - t1:.2f}s")
        debug_dataset_vs_reference(test_dl, fasta, context=args.context)
        return
    
    # --- MaxEntScan model ---
    if not args.skip_maxent:
        t_maxent = time.time()
        print("\n=== Starting MaxEntScan evaluation ===")
        maxent_test_dl = []
        for inputs, targets, M, coord in test_dl:
            # Use reference genome for padding
            batch_seq = [one_hot_to_sequence(seq) for seq in inputs]
            maxent_test_dl.append((inputs, targets, M, coord, batch_seq))

        results["MaxEntScan"], curves["MaxEntScan"] = evaluate_model_auprc_MaxEntScan(
            maxent_test_dl, return_curves=True
        )
        print(f"MaxEntScan evaluation completed in {time.time() - t_maxent:.2f}s")
        
        if not args.skip_plots:
            plot_pr_curves(curves, filename="maxentscan_auprc.pdf")
    
    # --- Load simple CNN model ---
    t_simple = time.time()
    model_path = os.path.join(args.save_dir, f"model.rep{args.num_replicates}.pth")
    if not os.path.exists(model_path):
        print(f"[Skip] {model_path} not found.")
        return

    print(f"\n=== Loading simple model {args.num_replicates} ===")
    model = SimpleNetModified_DA_SSE(input_channels=4).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Evaluate simple model
    results["Simple model"], curves["Simple model"] = evaluate_model_auprc_DLmodels(
        model, test_dl, args.num_replicates, return_curves=True
    )
    print(f"Simple model evaluation completed in {time.time() - t_simple:.2f}s")

    # --- SpliceAI models (lazy loaded only if needed) ---
    if not args.skip_spliceai:
        t_spliceai = time.time()
        print("\n=== Loading SpliceAI reference models ===")
        
        # Load model lazily with caching
        spliceai_model = load_spliceai_model_cached()
        spliceai_models = {"spliceai_1": spliceai_model}
        print(f"SpliceAI model loaded in {time.time() - t_spliceai:.2f}s")

        # --- Preload reference genome if needed ---
        fasta = None
        if args.mode == "with_ref_genome":
            if not args.reference_file:
                raise ValueError("Reference genome path must be provided in 'with_ref_genome' mode")
            print(f"Loading reference genome from {args.reference_file} ...")
            t_fasta = time.time()
            fasta = pyfastx.Fasta(args.reference_file)
            print(f"FASTA loaded in {time.time() - t_fasta:.2f}s")

        for name, model_ref in spliceai_models.items():
            print(f"\n=== Evaluating {name} ===")
            context = args.context

            adj_test_dl = []
            t_prep = time.time()
            for inputs, targets, M, coord in test_dl:
                if args.mode == "normal":
                    adj_inputs = prepare_input_for_spliceai(inputs, context)
                else:
                    # Use reference genome for padding
                    adj_inputs = prepare_input_with_reference(inputs, coord, fasta, context)
                adj_test_dl.append((adj_inputs, targets, M, coord))
            print(f"Data preparation for {name}: {time.time() - t_prep:.2f}s")

            t_eval = time.time()
            results[name], curves[name] = evaluate_model_auprc_DLmodels(
                model_ref, adj_test_dl, replicate_num=0, 
                return_curves=True, is_spliceai=True, model_name=name
            )
            print(f"Evaluation of {name} completed in {time.time() - t_eval:.2f}s")

    # --- Plot PR curves ---
    if args.plot and not args.skip_plots:
        print("\n=== Plotting PR curves ===")
        t_plot = time.time()
        plot_pr_curves(curves, filename=args.output_file)
        print(f"Plotting completed in {time.time() - t_plot:.2f}s")

    # Print summary
    print("\n=== Evaluation Summary ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")
    print("Evaluation complete.")

if __name__ == "__main__":
    main()