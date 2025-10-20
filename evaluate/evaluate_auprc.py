#!/usr/bin/env python3
"""
evaluate_auprc_replicates.py
----------------------------------
Evaluate trained replicate models on test data using auPRC (average precision).

Usage:
    python evaluate_auprc_replicates.py \
        --test_data ../../create_dataset/dataset_test_1.h5 \
        --save_dir ./trained_models \
        --replicate_id 12 \
        --batch_size 32 \
        --num_workers 4
"""

import os
import argparse
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from torch.utils import data
from tqdm import tqdm
import torch
from torch_fftconv import FFTConv1d
from torch import nn
from torch.utils import data
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
import pandas as pd


from models import SimpleNetModified_DA, H5Dataset


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


# -------------------------------
# Main evaluation routine
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate replicate models by auPRC.")
    parser.add_argument("--test_data", required=True, help="Path to test dataset (.h5)")
    parser.add_argument("--save_dir", required=True, help="Directory with saved model.rep*.pth files")
    parser.add_argument("--replicate_id", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot", action="store_true", default=True, help="Generate auPRC plot (default: True)")
    args = parser.parse_args()

    # Load test dataset
    print(f"Loading test data from {args.test_data}")
    test_dataset = H5Dataset(args.test_data)
    test_dl = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    
    model_path = os.path.join(args.save_dir, f"model.rep{args.replicate_id}.pth")
    if not os.path.exists(model_path):
        print(f"[Skip] {model_path} not found.")
        return

    print(f"\n=== Evaluating replicate {args.replicate_id} ===")
    model = SimpleNetModified_DA(input_channels=4).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    evaluate_model_auprc(model, test_dl, args.replicate_id)

    
    print("\nEvaluation complete")


if __name__ == "__main__":
    main()