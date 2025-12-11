import os
import sys
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import numpy as np
import torch

# -------------------------------
NUCLEOTIDES = {'A':0, 'C':1, 'G':2, 'T':3, 'N':-1}

def one_hot_encode_sequence(seq):
    """One-hot encode a DNA sequence string into shape (4, L)"""
    L = len(seq)
    x = np.zeros((4, L), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in NUCLEOTIDES and NUCLEOTIDES[base] >= 0:
            x[NUCLEOTIDES[base], i] = 1.0
    return x

def one_hot_to_sequence(one_hot_array):
    """Convert (4, L) one-hot tensor/array to DNA string"""
    if isinstance(one_hot_array, torch.Tensor):
        one_hot_array = one_hot_array.numpy()
    indices = np.argmax(one_hot_array, axis=0)
    seq = ''.join(['ACGT'[i] if np.sum(one_hot_array[:, j]) > 0 else 'N' 
                   for j, i in enumerate(indices)])
    return seq

# -------------------------------
# Plotting function
# -------------------------------
def plot_auprc_results(all_results, save_dir):
    """Plot auPRC results for all replicates using line plots"""
    if not all_results:
        print("No results to plot.")
        return
    
    # Extract data for plotting
    replicates = list(range(len(all_results)))
    donor_scores = [result["task1_per_class"][0] for result in all_results]
    acceptor_scores = [result["task1_per_class"][1] for result in all_results]
    mean_scores = [result["task1_mean"] for result in all_results]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot lines with markers
    plt.plot(replicates, donor_scores, marker='o', linewidth=2.5, markersize=8, 
             label='Donor auPRC', color='royalblue', alpha=0.8)
    plt.plot(replicates, acceptor_scores, marker='s', linewidth=2.5, markersize=8, 
             label='Acceptor auPRC', color='crimson', alpha=0.8)
    plt.plot(replicates, mean_scores, marker='^', linewidth=2.5, markersize=8, 
             label='Mean auPRC', color='forestgreen', alpha=0.8)
    
    plt.xlabel('Replicate Number')
    plt.ylabel('auPRC Score')
    plt.title('auPRC Performance Across Replicates')
    plt.xticks(replicates, [f'Rep {i}' for i in replicates])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, "auprc_replicates_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {plot_path}")

def plot_regression_results(all_results, save_dir):
    """
    Plot regression metrics (MSE, R2, Pearson, Spearman) across replicates.
    Clean integer x-axis (no .5 ticks).
    """
    if not all_results:
        print("No results to plot.")
        return
    
    replicates = list(range(1, len(all_results) + 1))
    
    # Extract metrics
    mse_scores = [result["Combined"]["MSE"] for result in all_results]
    r2_scores = [result["Combined"]["R2"] for result in all_results]
    pearson_scores = [result["Combined"]["Pearson"] for result in all_results]
    spearman_scores = [result["Combined"]["Spearman"] for result in all_results]
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(replicates, mse_scores, marker='o', linewidth=2.5, markersize=8,
             label='MSE ↓', color='crimson', alpha=0.8)
    plt.plot(replicates, r2_scores, marker='s', linewidth=2.5, markersize=8,
             label='R² ↑', color='royalblue', alpha=0.8)
    plt.plot(replicates, pearson_scores, marker='^', linewidth=2.5, markersize=8,
             label='Pearson ↑', color='forestgreen', alpha=0.8)
    plt.plot(replicates, spearman_scores, marker='v', linewidth=2.5, markersize=8,
             label='Spearman ↑', color='goldenrod', alpha=0.8)

    # --- Clean integer ticks ---
    plt.xticks(replicates, [f"Rep {r}" for r in replicates])
    plt.xlim(min(replicates) - 0.5, max(replicates) + 0.5)
    
    plt.xlabel("Replicate Number")
    plt.ylabel("Metric Value")
    plt.title("Regression Performance Across Replicates")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "regression_replicates_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to: {plot_path}")

# Training functions
def progress_bar(batch_idx, total_batches, message):
    bar_length = 20
    progress = float(batch_idx) / total_batches
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] {1:.1%} {2}".format(
        "=" * block + "-" * (bar_length - block), progress, message)
    sys.stdout.write(text)
    sys.stdout.flush()


# --------------------------
# evaluation functions
# --------------------------
def evaluate_model_auprc(model, test_dl, replicate_num):
    """
    Evaluate model and compute auPRC for Donor and Acceptor channels with masking
    """
    model.eval()
    all_outputs = []
    all_targets = []
    all_masks = []
    
    print(f"\n=== Evaluating Replicate {replicate_num} on Test Set ===")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, M, coord) in enumerate(test_dl):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(test_dl)}")
                
            inputs, targets, M = inputs.cuda(), targets.cuda(), M.cuda()
            outputs = model(inputs)
            
            # Store predictions, targets, and masks
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(M.cpu().numpy())
    
    # Concatenate all batches
    all_outputs = np.concatenate(all_outputs, axis=0)  # shape: (N, 2, 4000)
    all_targets = np.concatenate(all_targets, axis=0)  # shape: (N, 2, 4000)
    all_masks = np.concatenate(all_masks, axis=0)      # shape: (N, 4000)
    
    print(f"Collected predictions: {all_outputs.shape}")
    print(f"Collected targets: {all_targets.shape}")
    print(f"Collected masks: {all_masks.shape}")
    
    # Calculate auPRC for each channel
    auprc_results = {}
    
    for channel_idx, channel_name in enumerate(['Donor', 'Acceptor']):
        print(f"\nComputing auPRC for {channel_name} channel...")
        
        # Get predictions and targets for this channel
        y_scores = all_outputs[:, channel_idx, :]  # shape: (N, 4000)
        y_true = all_targets[:, channel_idx, :]    # shape: (N, 4000)
        
        # Apply mask: only keep positions where mask == 1
        valid_positions = all_masks == 1
        
        # Flatten only the valid positions
        y_scores_valid = y_scores[valid_positions]
        y_true_valid = y_true[valid_positions]
        
        print(f"  Total positions: {np.prod(y_true.shape)}")
        print(f"  Valid positions: {len(y_true_valid)}")
        print(f"  Masked positions: {np.prod(y_true.shape) - len(y_true_valid)}")
        print(f"  Positive samples: {np.sum(y_true_valid == 1)}")
        print(f"  Negative samples: {np.sum(y_true_valid == 0)}")
        print(f"  Positive rate: {np.sum(y_true_valid == 1) / len(y_true_valid):.6f}")
        
        # Only compute auPRC if we have both positive and negative samples
        if len(np.unique(y_true_valid)) < 2:
            print(f"  WARNING: Only one class present in {channel_name} channel")
            auprc_results[channel_name] = 0.0
            continue
            
        # Calculate precision-recall curve and auPRC
        precision, recall, thresholds = precision_recall_curve(y_true_valid, y_scores_valid)
        auprc = auc(recall, precision)
        
        auprc_results[channel_name] = auprc
        print(f"  {channel_name} auPRC: {auprc:.6f}")
    
    # Calculate combined auPRC (micro-average across both channels)
    print(f"\nComputing combined auPRC...")
    
    # Method 1: Process each channel separately and combine
    combined_scores = []
    combined_labels = []
    
    for channel_idx in range(2):
        y_scores = all_outputs[:, channel_idx, :]  # shape: (N, 4000)
        y_true = all_targets[:, channel_idx, :]    # shape: (N, 4000)
        valid_positions = all_masks == 1
        
        # Flatten only the valid positions for this channel
        y_scores_valid = y_scores[valid_positions]
        y_true_valid = y_true[valid_positions]
        
        combined_scores.append(y_scores_valid)
        combined_labels.append(y_true_valid)
    
    # Concatenate across both channels
    y_scores_combined_valid = np.concatenate(combined_scores)
    y_true_combined_valid = np.concatenate(combined_labels)
    
    print(f"  Combined valid positions: {len(y_true_combined_valid)}")
    print(f"  Combined positive samples: {np.sum(y_true_combined_valid == 1)}")
    print(f"  Combined negative samples: {np.sum(y_true_combined_valid == 0)}")
    print(f"  Combined positive rate: {np.sum(y_true_combined_valid == 1) / len(y_true_combined_valid):.6f}")
    
    if len(np.unique(y_true_combined_valid)) >= 2:
        precision_combined, recall_combined, _ = precision_recall_curve(y_true_combined_valid, y_scores_combined_valid)
        auprc_combined = auc(recall_combined, precision_combined)
        auprc_results['Combined'] = auprc_combined
        print(f"Combined auPRC: {auprc_combined:.6f}")
    else:
        print(f"WARNING: Only one class present in combined data")
        auprc_results['Combined'] = 0.0
    
    return auprc_results

def evaluate_model_regression(model, test_dl, replicate_num):
    """
    Evaluate model performance on SSE (continuous) labels using regression metrics,
    ignoring TN positions (i.e., sites where y_true == 0 for both channels).

    Metrics computed: MSE, MAE, R², Pearson, Spearman.
    """
    model.eval()
    all_outputs, all_targets = [], []

    print(f"\n=== Evaluating Replicate {replicate_num} (Regression metrics, TN ignored) ===")

    with torch.no_grad():
        for batch_idx, (inputs, targets, _, _) in enumerate(test_dl):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(test_dl)}")

            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_outputs = np.concatenate(all_outputs, axis=0)  # (N, 2, L)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 2, L)
    print(f"Collected predictions: {all_outputs.shape}, targets: {all_targets.shape}")

    results = {}

    # -----------------------------
    # Per-channel metrics
    # -----------------------------
    for channel_idx, channel_name in enumerate(["Donor", "Acceptor"]):
        y_pred = all_outputs[:, channel_idx, :]
        y_true = all_targets[:, channel_idx, :]

        # Only consider positions with non-zero target (ignore TN)
        valid_positions = y_true != 0
        y_pred_valid = y_pred[valid_positions]
        y_true_valid = y_true[valid_positions]

        if len(y_true_valid) == 0:
            print(f"  WARNING: No valid positions found for {channel_name}")
            continue

        results[channel_name] = {
            "MSE": mean_squared_error(y_true_valid, y_pred_valid),
            "MAE": mean_absolute_error(y_true_valid, y_pred_valid),
            "R2": r2_score(y_true_valid, y_pred_valid),
            "Pearson": pearsonr(y_true_valid, y_pred_valid)[0],
            "Spearman": spearmanr(y_true_valid, y_pred_valid)[0],
        }

        print(f"\n{channel_name} metrics:")
        for metric, value in results[channel_name].items():
            print(f"  {metric}: {value:.6f}")

    # -----------------------------
    # Combined metrics across both channels
    # -----------------------------
    valid_positions = all_targets != 0
    combined_pred = all_outputs[valid_positions]
    combined_true = all_targets[valid_positions]

    results["Combined"] = {
        "MSE": mean_squared_error(combined_true, combined_pred),
        "MAE": mean_absolute_error(combined_true, combined_pred),
        "R2": r2_score(combined_true, combined_pred),
        "Pearson": pearsonr(combined_true, combined_pred)[0],
        "Spearman": spearmanr(combined_true, combined_pred)[0],
    }

    print("\nCombined metrics (both channels, TN ignored):")
    for metric, value in results["Combined"].items():
        print(f"  {metric}: {value:.6f}")

    return results
