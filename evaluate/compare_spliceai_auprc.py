import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from sklearn.metrics import precision_recall_curve, auc
from models import SimpleNetModified_DA, H5Dataset
from tensorflow.keras.models import load_model
import tensorflow as tf

# -------------------------------
# Evaluate model (supports both output formats)
# -------------------------------
def evaluate_model_auprc(model, test_dl, replicate_num, return_curves=False, is_spliceai=False, model_name=""):
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
    Remove 500bp padding on each side (from 5000→4000),
    then pad with N's (0,0,0,0) according to model context window.
    """
    x_core = one_hot_input  # (B, 4, 5000)
    pad_len = model_context // 2
    pad = torch.zeros((x_core.shape[0], 4, pad_len), dtype=x_core.dtype)
    return torch.cat([pad, x_core, pad], dim=-1)  # (B, 4, 5000 + context)


# -------------------------------
# Plot PR curves
# -------------------------------
def plot_pr_curves(curves_dict):
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

        ax.set_title(f"{ch} Precision–Recall Curve", fontsize=14)
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
    plt.savefig("model_auprc_comparison.pdf", dpi=300, format='pdf')


# -------------------------------
# Main evaluation routine
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate replicate models by auPRC.")
    parser.add_argument("--test_data", required=True, help="Path to test dataset (.h5)")
    parser.add_argument("--save_dir", required=True, help="Directory with saved model.rep*.pth files")
    parser.add_argument("--num_replicates", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot", action="store_true", default=True, help="Generate auPRC plot (default: True)")
    args = parser.parse_args()

    print(f"Loading test data from {args.test_data}")
    test_dataset = H5Dataset(args.test_data)
    test_dl = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    
    # --- Your trained model ---
    model_path = os.path.join(args.save_dir, f"model.rep{args.num_replicates}.pth")
    if not os.path.exists(model_path):
        print(f"[Skip] {model_path} not found.")
        return

    print(f"\n=== Evaluating replicate {args.num_replicates} ===")
    model = SimpleNetModified_DA(input_channels=4).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results, curves = {}, {}

    # Evaluate your model
    results["Current"], curves["Current"] = evaluate_model_auprc(model, test_dl, args.num_replicates, return_curves=True)

    # --- SpliceAI models ---
    print("\n=== Loading SpliceAI reference models ===")
    spliceai_models = {
        "spliceai_1": load_model('../../SpliceAI/spliceai/models/spliceai1.h5', compile=False),
    }

    for name, model_ref in spliceai_models.items():
        print(f"\n=== Evaluating {name} ===")
        context = 10000

        adj_test_dl = []
        for inputs, targets, M, coord in test_dl:
            adj_inputs = prepare_input_for_spliceai(inputs, context)
            adj_test_dl.append((adj_inputs, targets, M, coord))

        results[name], curves[name] = evaluate_model_auprc(
            model_ref, adj_test_dl, replicate_num=0, return_curves=True, is_spliceai=True, model_name=name
        )

    # --- Plot true PR curves ---
    if args.plot:
        print("\n=== Plotting true PR curves ===")
        plot_pr_curves(curves)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
